// Update Clock
setInterval(() => {
    const now = new Date();
    const time = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    const date = now.toLocaleDateString('vi-VN');
    document.getElementById('clock').innerText = `${time} - ${date}`;
}, 1000);

// Polling Status from Server
let isProcessing = false;

function checkStatus() {
    if (isProcessing) return;

    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'CONFIRM' && data.data) {
                showResult(data.data);
                isProcessing = true;
            } else if (data.status === 'PROCESSING') {
                showProcessing(data.progress);
            } else {
                showIdle();
            }
        })
        .catch(err => console.error(err));
}

// Poll every 300ms
const poller = setInterval(checkStatus, 300);

function showProcessing(progress) {
    // Ẩn bảng kết quả, hiện bảng Idle (nhưng đổi text status)
    document.getElementById('info-idle').classList.remove('hidden');
    document.getElementById('info-result').classList.add('hidden');

    // Hiệu ứng Loading (Xoay vòng)
    const oval = document.querySelector('.oval-overlay');
    oval.classList.add('processing');
    oval.classList.remove('success');

    // Cập nhật Status Bar
    const statusEl = document.getElementById('scan-status');
    statusEl.innerHTML = `<span class="iconify spinning" data-icon="mdi:loading"></span> Đang xử lý... ${Math.round(progress)}%`;
    statusEl.style.background = "#F26F21";
}

function showResult(student) {
    document.getElementById('info-idle').classList.add('hidden');
    document.getElementById('info-result').classList.remove('hidden');

    document.getElementById('st-name').innerText = student.name;
    document.getElementById('st-id').innerText = `MSSV: ${student.student_id}`;
    document.getElementById('st-schedule').innerText = student.schedule;
    document.getElementById('st-room').innerText = student.room;

    // Hiệu ứng Thành công (Xanh lá)
    const oval = document.querySelector('.oval-overlay');
    oval.classList.remove('processing');
    oval.classList.add('success');

    const statusEl = document.getElementById('scan-status');
    statusEl.innerHTML = '<span class="iconify" data-icon="mdi:check-circle"></span> Đã nhận diện!';
    statusEl.style.background = "#2ECC71";
}

function showIdle() {
    document.getElementById('info-idle').classList.remove('hidden');
    document.getElementById('info-result').classList.add('hidden');

    // Reset Trạng thái
    const oval = document.querySelector('.oval-overlay');
    oval.classList.remove('processing', 'success');

    const statusEl = document.getElementById('scan-status');
    statusEl.innerHTML = '<span class="dot blinking"></span> Đang quét...';
    statusEl.style.background = "rgba(0,0,0,0.8)";
}

// Handle Buttons
function handleAction(action) {
    fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: action })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                isProcessing = false; // Resume polling
                showIdle();
            }
        });
}
