// Copyright 2026 Project A8 Authors
// SPDX-License-Identifier: BSD-3-Clause
//
// Iceoryx2-based virtio-media capture device for EVS camera support.
//
// This device receives camera frames via iceoryx2 shared memory and exposes them
// as a V4L2 capture device to the guest. This enables scalable camera virtualization
// without requiring kernel modules like v4l2loopback.

use std::collections::VecDeque;
use std::io::Result as IoResult;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::os::fd::AsFd;
use std::os::fd::BorrowedFd;
use std::time::Duration;
use std::time::Instant;

use base::AsRawDescriptor;
use base::Timer;
use base::TimerTrait;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::*;
use iceoryx2::service::ipc_threadsafe;

use virtio_media::ioctl::virtio_media_dispatch_ioctl;
use virtio_media::ioctl::IoctlResult;
use virtio_media::ioctl::VirtioMediaIoctlHandler;
use virtio_media::memfd::MemFdBuffer;
use virtio_media::mmap::MmapMappingManager;
use virtio_media::protocol::DequeueBufferEvent;
use virtio_media::protocol::SgEntry;
use virtio_media::protocol::V4l2Event;
use virtio_media::protocol::V4l2Ioctl;
use virtio_media::protocol::VIRTIO_MEDIA_MMAP_FLAG_RW;
use virtio_media::v4l2r::bindings;
use virtio_media::v4l2r::bindings::v4l2_fmtdesc;
use virtio_media::v4l2r::bindings::v4l2_format;
use virtio_media::v4l2r::bindings::v4l2_pix_format;
use virtio_media::v4l2r::bindings::v4l2_requestbuffers;
use virtio_media::v4l2r::ioctl::BufferCapabilities;
use virtio_media::v4l2r::ioctl::BufferField;
use virtio_media::v4l2r::ioctl::BufferFlags;
use virtio_media::v4l2r::ioctl::V4l2Buffer;
use virtio_media::v4l2r::ioctl::V4l2PlanesWithBackingMut;
use virtio_media::v4l2r::memory::MemoryType;
use virtio_media::v4l2r::PixelFormat;
use virtio_media::v4l2r::QueueType;
use virtio_media::io::ReadFromDescriptorChain;
use virtio_media::io::WriteToDescriptorChain;
use virtio_media::VirtioMediaDevice;
use virtio_media::VirtioMediaDeviceSession;
use virtio_media::VirtioMediaEventQueue;
use virtio_media::VirtioMediaHostMemoryMapper;

/// Pixel format type for iceoryx2 camera
///
/// Note: Android Automotive EVS HAL (trout-stable) only supports specific V4L2 formats:
/// - V4L2_PIX_FMT_ARGB32 (BA24) for RGB - not BGRA32/AB24
/// - V4L2_PIX_FMT_NV21 for YUV - not NV12
/// This enum maps to EVS-compatible formats internally.
#[derive(Clone, Copy, Debug, Default)]
pub enum PixelFormatType {
    /// RGBA 32-bit - maps to V4L2_PIX_FMT_ARGB32 (BA24) for EVS compatibility
    #[default]
    Rgba,
    /// YUV420 semi-planar - maps to NV21 for EVS compatibility (EVS doesn't support NV12)
    Nv12,
    /// NV21 YUV420 semi-planar (Android's native YUV format)
    Nv21,
}

/// Camera frame format configuration
#[derive(Clone, Debug)]
pub struct FrameConfig {
    pub width: u32,
    pub height: u32,
    pub pixel_format: u32,
    pub bytes_per_line: u32,
    pub buffer_size: u32,
}

impl Default for FrameConfig {
    fn default() -> Self {
        // Default to RGBA format at 640x480 (best for Android EVS)
        Self::new(640, 480, PixelFormatType::Rgba)
    }
}

impl FrameConfig {
    /// Create a new frame config with the specified dimensions and format
    ///
    /// Note: The EVS HAL in Android Automotive (trout-stable) only supports specific formats:
    /// - V4L2_PIX_FMT_ARGB32 (BA24), not BGRA32 (AB24)
    /// - V4L2_PIX_FMT_NV21, not NV12
    /// - V4L2_PIX_FMT_YUYV, BGRX32, RGB32, RGB24, etc.
    pub fn new(width: u32, height: u32, format: PixelFormatType) -> Self {
        match format {
            PixelFormatType::Rgba => Self {
                width,
                height,
                // Use ARGB32 (BA24) which EVS HAL supports, not BGRA32 (AB24)
                pixel_format: PixelFormat::from_fourcc(b"BA24").to_u32(), // V4L2_PIX_FMT_ARGB32
                bytes_per_line: width * 4,
                buffer_size: width * height * 4,
            },
            PixelFormatType::Nv12 => Self {
                width,
                height,
                // EVS HAL supports NV21, not NV12 - using NV21 for compatibility
                pixel_format: PixelFormat::from_fourcc(b"NV21").to_u32(),
                bytes_per_line: width,
                buffer_size: width * height * 3 / 2,
            },
            PixelFormatType::Nv21 => Self {
                width,
                height,
                pixel_format: PixelFormat::from_fourcc(b"NV21").to_u32(),
                bytes_per_line: width,
                buffer_size: width * height * 3 / 2,
            },
        }
    }

    pub fn rgba(width: u32, height: u32) -> Self {
        Self::new(width, height, PixelFormatType::Rgba)
    }

    pub fn nv12(width: u32, height: u32) -> Self {
        Self::new(width, height, PixelFormatType::Nv12)
    }

    pub fn nv21(width: u32, height: u32) -> Self {
        Self::new(width, height, PixelFormatType::Nv21)
    }
}

/// Frame data received from iceoryx2
#[derive(Debug, Clone, ZeroCopySend)]
#[repr(C)]
pub struct CameraFrame {
    pub timestamp_ns: u64,
    pub sequence: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub data_len: u32,
    // Frame data follows (up to max frame size)
    pub data: [u8; 1920 * 1080 * 4], // Max 4K RGBA
}

impl Default for CameraFrame {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            sequence: 0,
            width: 640,
            height: 480,
            format: 0,
            data_len: 0,
            data: [0u8; 1920 * 1080 * 4],
        }
    }
}

/// Buffer state tracking
#[derive(Debug, PartialEq, Eq)]
enum BufferState {
    New,
    Incoming,
    Outgoing { sequence: u32 },
}

/// Single buffer information
struct Buffer {
    state: BufferState,
    v4l2_buffer: V4l2Buffer,
    fd: MemFdBuffer,
    offset: u32,
}

impl Buffer {
    fn set_state(&mut self, state: BufferState, buffer_size: u32) {
        let mut flags = self.v4l2_buffer.flags();
        match state {
            BufferState::New => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = 0;
                flags -= BufferFlags::QUEUED;
            }
            BufferState::Incoming => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = 0;
                flags |= BufferFlags::QUEUED;
            }
            BufferState::Outgoing { sequence } => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = buffer_size;
                self.v4l2_buffer.set_sequence(sequence);
                self.v4l2_buffer.set_timestamp(bindings::timeval {
                    tv_sec: (sequence as i64) / 30, // Assume 30fps
                    tv_usec: ((sequence as i64) % 30) * 33333,
                });
                flags -= BufferFlags::QUEUED;
            }
        }
        self.v4l2_buffer.set_flags(flags);
        self.state = state;
    }
}

/// Session data for iceoryx2 capture device
pub struct Iceoryx2CaptureSession {
    id: u32,
    frame_count: u64,
    buffers: Vec<Buffer>,
    queued_buffers: VecDeque<usize>,
    streaming: bool,
    poll_timer: Timer,
    poll_timer_armed: bool,
}

impl VirtioMediaDeviceSession for Iceoryx2CaptureSession {
    fn poll_fd(&self) -> Option<BorrowedFd> {
        // SAFETY: `poll_timer` is owned by this session and lives at least as long as `&self`.
        Some(unsafe { BorrowedFd::borrow_raw(self.poll_timer.as_raw_descriptor()) })
    }
}

impl Iceoryx2CaptureSession {
    const POLL_INTERVAL: Duration = Duration::from_millis(10);

    fn update_poll_timer(&mut self) -> Result<(), i32> {
        let should_arm = self.streaming && !self.queued_buffers.is_empty();

        if should_arm && !self.poll_timer_armed {
            self.poll_timer
                .reset_repeating(Self::POLL_INTERVAL)
                .map_err(|_| libc::EIO)?;
            self.poll_timer_armed = true;
        } else if !should_arm && self.poll_timer_armed {
            self.poll_timer.clear().map_err(|_| libc::EIO)?;
            self.poll_timer_armed = false;
        }

        Ok(())
    }

    /// Process a single queued buffer with the provided frame data.
    /// Returns true if a buffer was processed, false if no buffers queued.
    fn process_one_buffer<Q: VirtioMediaEventQueue>(
        &mut self,
        evt_queue: &mut Q,
        frame_config: &FrameConfig,
        frame_data: &[u8],
    ) -> IoctlResult<bool> {
        let buf_id = match self.queued_buffers.pop_front() {
            Some(id) => id,
            None => return Ok(false),
        };

        let buffer = self.buffers.get_mut(buf_id).ok_or(libc::EIO)?;
        let sequence = self.frame_count as u32;

        // Write frame data to buffer
        buffer
            .fd
            .as_file()
            .seek(SeekFrom::Start(0))
            .map_err(|_| libc::EIO)?;
        buffer
            .fd
            .as_file()
            .write_all(frame_data)
            .map_err(|_| libc::EIO)?;

        buffer.set_state(BufferState::Outgoing { sequence }, frame_config.buffer_size);
        self.frame_count += 1;

        let v4l2_buffer = buffer.v4l2_buffer.clone();
        evt_queue.send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
            self.id,
            v4l2_buffer,
        )));

        Ok(true)
    }
}

/// Iceoryx2-based capture device for virtio-media
pub struct Iceoryx2CaptureDevice<Q: VirtioMediaEventQueue, HM: VirtioMediaHostMemoryMapper> {
    evt_queue: Q,
    mmap_manager: MmapMappingManager<HM>,
    active_session: Option<u32>,
    topic_name: String,
    frame_config: FrameConfig,
    // Iceoryx2 subscriber (lazily initialized)
    subscriber: Option<Subscriber<ipc_threadsafe::Service, CameraFrame, ()>>,
    node: Option<Node<ipc_threadsafe::Service>>,
    last_subscriber_init_attempt: Option<Instant>,
}

impl<Q, HM> Iceoryx2CaptureDevice<Q, HM>
where
    Q: VirtioMediaEventQueue,
    HM: VirtioMediaHostMemoryMapper,
{
    pub fn new(
        evt_queue: Q,
        mapper: HM,
        topic_name: String,
        frame_config: FrameConfig,
    ) -> Self {
        Self {
            evt_queue,
            mmap_manager: MmapMappingManager::from(mapper),
            active_session: None,
            topic_name,
            frame_config,
            subscriber: None,
            node: None,
            last_subscriber_init_attempt: None,
        }
    }

    fn maybe_init_subscriber(&mut self) {
        const RETRY_INTERVAL: Duration = Duration::from_secs(1);

        if self.subscriber.is_some() {
            return;
        }

        if let Some(last) = self.last_subscriber_init_attempt {
            if last.elapsed() < RETRY_INTERVAL {
                return;
            }
        }
        self.last_subscriber_init_attempt = Some(Instant::now());

        match self.init_subscriber() {
            Ok(()) => (),
            Err(e) => {
                base::debug!("iceoryx2 subscriber init not ready: {}", e);
            }
        }
    }

    /// Initialize iceoryx2 subscriber (best effort, non-critical)
    /// If initialization fails, we will retry later.
    fn init_subscriber(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.subscriber.is_some() {
            return Ok(());
        }

        // Only try to connect to existing service, don't create
        // This avoids blocking on service creation if publisher isn't ready
        let node = NodeBuilder::new()
            .name(&NodeName::new(&format!("crosvm-evs-{}", std::process::id()))?)
            .create::<ipc_threadsafe::Service>()?;

        // Try to open existing service first (non-blocking)
        let service_name = ServiceName::new(&self.topic_name)?;
        let service = match node
            .service_builder(&service_name)
            .publish_subscribe::<CameraFrame>()
            .open()
        {
            Ok(svc) => svc,
            Err(_) => {
                return Err("Service not available yet".into());
            }
        };

        let subscriber = service.subscriber_builder().create()?;

        self.node = Some(node);
        self.subscriber = Some(subscriber);

        base::info!(
            "Iceoryx2 capture device initialized, subscribing to topic: {}",
            self.topic_name
        );

        Ok(())
    }

    /// Try to receive and process frames from iceoryx2 (non-blocking).
    /// Processes all available frames from the subscriber, filling queued buffers.
    /// Returns number of frames processed.
    fn try_process_frames(&mut self, session: &mut Iceoryx2CaptureSession) -> usize {
        if !session.streaming || session.queued_buffers.is_empty() {
            return 0;
        }

        // Lazily initialize subscriber (non-blocking attempt, throttled).
        self.maybe_init_subscriber();

        let mut processed = 0;

        // Try to receive real frames first (non-blocking)
        if let Some(sub) = self.subscriber.as_ref() {
            loop {
                if session.queued_buffers.is_empty() {
                    break;
                }

                match sub.receive() {
                    Ok(Some(sample)) => {
                        // Access frame data directly without cloning the 8MB struct
                        let payload = sample.payload();
                        let data_len = payload.data_len as usize;
                        if data_len > payload.data.len() {
                            base::warn!(
                                "Dropping invalid frame: data_len {} exceeds payload buffer {}",
                                data_len,
                                payload.data.len()
                            );
                            continue;
                        }
                        if data_len > self.frame_config.buffer_size as usize {
                            base::warn!(
                                "Dropping oversized frame: data_len {} exceeds buffer_size {}",
                                data_len,
                                self.frame_config.buffer_size
                            );
                            continue;
                        }
                        let frame_data = &payload.data[..data_len];

                        match session.process_one_buffer(
                            &mut self.evt_queue,
                            &self.frame_config,
                            frame_data,
                        ) {
                            Ok(true) => {
                                processed += 1;
                            }
                            Ok(false) => break,
                            Err(e) => {
                                base::warn!("Error processing buffer: {}", e);
                                break;
                            }
                        }
                    }
                    Ok(None) => {
                        // No more frames available
                        break;
                    }
                    Err(e) => {
                        base::warn!("Error receiving frame: {:?}", e);
                        break;
                    }
                }
            }
        }

        processed
    }
}

const INPUTS: [bindings::v4l2_input; 1] = [bindings::v4l2_input {
    index: 0,
    name: *b"iceoryx2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
    type_: bindings::V4L2_INPUT_TYPE_CAMERA,
    ..unsafe { std::mem::zeroed() }
}];

impl<Q, HM, Reader, Writer> VirtioMediaDevice<Reader, Writer> for Iceoryx2CaptureDevice<Q, HM>
where
    Q: VirtioMediaEventQueue,
    HM: VirtioMediaHostMemoryMapper,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    type Session = Iceoryx2CaptureSession;

    fn new_session(&mut self, session_id: u32) -> Result<Self::Session, i32> {
        Ok(Iceoryx2CaptureSession {
            id: session_id,
            frame_count: 0,
            buffers: Vec::new(),
            queued_buffers: VecDeque::new(),
            streaming: false,
            poll_timer: Timer::new().map_err(|_| libc::EIO)?,
            poll_timer_armed: false,
        })
    }

    fn close_session(&mut self, session: Self::Session) {
        if self.active_session == Some(session.id) {
            self.active_session = None;
        }
        for buffer in &session.buffers {
            self.mmap_manager.unregister_buffer(buffer.offset);
        }
    }

    fn do_ioctl(
        &mut self,
        session: &mut Self::Session,
        ioctl: V4l2Ioctl,
        reader: &mut Reader,
        writer: &mut Writer,
    ) -> IoResult<()> {
        virtio_media_dispatch_ioctl(self, session, ioctl, reader, writer)
    }

    fn do_mmap(
        &mut self,
        session: &mut Self::Session,
        flags: u32,
        offset: u32,
    ) -> Result<(u64, u64), i32> {
        let buffer = session
            .buffers
            .iter_mut()
            .find(|b| b.offset == offset)
            .ok_or(libc::EINVAL)?;
        let rw = (flags & VIRTIO_MEDIA_MMAP_FLAG_RW) != 0;
        let fd = buffer.fd.as_file().as_fd();
        self.mmap_manager
            .create_mapping(offset, fd, rw)
            .map_err(|_| libc::EINVAL)
    }

    fn do_munmap(&mut self, guest_addr: u64) -> Result<(), i32> {
        self.mmap_manager
            .remove_mapping(guest_addr)
            .map(|_| ())
            .map_err(|_| libc::EINVAL)
    }

    fn process_events(&mut self, session: &mut Self::Session) -> Result<(), i32> {
        // Drain the timerfd to avoid busy looping in the wait context.
        session.poll_timer.mark_waited().map_err(|_| libc::EIO)?;

        self.try_process_frames(session);
        session.update_poll_timer()?;

        Ok(())
    }
}

impl<Q, HM> VirtioMediaIoctlHandler for Iceoryx2CaptureDevice<Q, HM>
where
    Q: VirtioMediaEventQueue,
    HM: VirtioMediaHostMemoryMapper,
{
    type Session = Iceoryx2CaptureSession;

    fn enum_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        if index > 0 {
            return Err(libc::EINVAL);
        }

        Ok(v4l2_fmtdesc {
            index: 0,
            type_: queue as u32,
            pixelformat: self.frame_config.pixel_format,
            ..Default::default()
        })
    }

    fn g_fmt(&mut self, _session: &Self::Session, queue: QueueType) -> IoctlResult<v4l2_format> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }

        let pix = v4l2_pix_format {
            width: self.frame_config.width,
            height: self.frame_config.height,
            pixelformat: self.frame_config.pixel_format,
            field: bindings::v4l2_field_V4L2_FIELD_NONE,
            bytesperline: self.frame_config.bytes_per_line,
            sizeimage: self.frame_config.buffer_size,
            colorspace: bindings::v4l2_colorspace_V4L2_COLORSPACE_SRGB,
            ..Default::default()
        };

        Ok(v4l2_format {
            type_: queue as u32,
            fmt: bindings::v4l2_format__bindgen_ty_1 { pix },
        })
    }

    fn s_fmt(
        &mut self,
        _session: &mut Self::Session,
        queue: QueueType,
        _format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        // We only support our configured format
        self.g_fmt(_session, queue)
    }

    fn try_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        _format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        self.g_fmt(_session, queue)
    }

    fn reqbufs(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        memory: MemoryType,
        count: u32,
    ) -> IoctlResult<v4l2_requestbuffers> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        if memory != MemoryType::Mmap {
            return Err(libc::EINVAL);
        }
        if session.streaming {
            return Err(libc::EBUSY);
        }
        match self.active_session {
            Some(id) if id != session.id => return Err(libc::EBUSY),
            _ => (),
        }

        if count == 0 {
            self.active_session = None;
            self.streamoff(session, queue)?;
        } else {
            session.queued_buffers.clear();
            for buffer in session.buffers.iter_mut() {
                buffer.set_state(BufferState::New, self.frame_config.buffer_size);
            }
            self.active_session = Some(session.id);
        }

        let count = std::cmp::min(count, 32);

        for buffer in &session.buffers {
            self.mmap_manager.unregister_buffer(buffer.offset);
        }

        session.buffers = (0..count)
            .map(|i| {
                MemFdBuffer::new(self.frame_config.buffer_size as u64)
                    .map_err(|e| {
                        base::error!("failed to allocate MMAP buffers: {:#}", e);
                        libc::ENOMEM
                    })
                    .and_then(|fd| {
                        let offset = self
                            .mmap_manager
                            .register_buffer(None, self.frame_config.buffer_size)
                            .map_err(|_| libc::EINVAL)?;

                        let mut v4l2_buffer =
                            V4l2Buffer::new(QueueType::VideoCapture, i, MemoryType::Mmap);
                        if let V4l2PlanesWithBackingMut::Mmap(mut planes) =
                            v4l2_buffer.planes_with_backing_iter_mut()
                        {
                            let mut plane = planes.next().unwrap();
                            plane.set_mem_offset(offset);
                            *plane.length = self.frame_config.buffer_size;
                        } else {
                            panic!("Buffer type mismatch");
                        }
                        v4l2_buffer.set_field(BufferField::None);
                        v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);

                        Ok(Buffer {
                            state: BufferState::New,
                            v4l2_buffer,
                            fd,
                            offset,
                        })
                    })
            })
            .collect::<Result<_, _>>()?;

        Ok(v4l2_requestbuffers {
            count,
            type_: queue as u32,
            memory: memory as u32,
            capabilities: (BufferCapabilities::SUPPORTS_MMAP
                | BufferCapabilities::SUPPORTS_ORPHANED_BUFS)
                .bits(),
            flags: 0,
            reserved: [0; 3],
        })
    }

    fn querybuf(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<V4l2Buffer> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        let buffer = session.buffers.get(index as usize).ok_or(libc::EINVAL)?;
        Ok(buffer.v4l2_buffer.clone())
    }

    fn qbuf(
        &mut self,
        session: &mut Self::Session,
        buffer: V4l2Buffer,
        _guest_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<V4l2Buffer> {
        let host_buffer = session
            .buffers
            .get_mut(buffer.index() as usize)
            .ok_or(libc::EINVAL)?;

        if matches!(host_buffer.state, BufferState::Incoming) {
            return Err(libc::EINVAL);
        }

        host_buffer.set_state(BufferState::Incoming, self.frame_config.buffer_size);
        session.queued_buffers.push_back(buffer.index() as usize);

        let buffer = host_buffer.v4l2_buffer.clone();

        // Non-blocking: try to process any available frames
        if session.streaming {
            self.try_process_frames(session);
        }
        session.update_poll_timer()?;

        Ok(buffer)
    }

    fn streamon(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QueueType::VideoCapture || session.buffers.is_empty() {
            return Err(libc::EINVAL);
        }
        session.streaming = true;

        // Non-blocking: try to process any available frames
        // If no frames yet, buffers stay queued until frames arrive
        self.try_process_frames(session);
        session.update_poll_timer()?;

        Ok(())
    }

    fn streamoff(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        session.streaming = false;
        session.queued_buffers.clear();
        for buffer in session.buffers.iter_mut() {
            buffer.set_state(BufferState::New, self.frame_config.buffer_size);
        }
        session.update_poll_timer()?;
        Ok(())
    }

    fn g_input(&mut self, _session: &Self::Session) -> IoctlResult<i32> {
        Ok(0)
    }

    fn s_input(&mut self, _session: &mut Self::Session, input: i32) -> IoctlResult<i32> {
        if input != 0 {
            Err(libc::EINVAL)
        } else {
            Ok(0)
        }
    }

    fn enuminput(
        &mut self,
        _session: &Self::Session,
        index: u32,
    ) -> IoctlResult<bindings::v4l2_input> {
        match INPUTS.get(index as usize) {
            Some(&input) => Ok(input),
            None => Err(libc::EINVAL),
        }
    }
}
