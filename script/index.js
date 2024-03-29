// UI BEGIN
const video = document.getElementById('video');
const videoControls = document.getElementById('video-controls');
const videoContainer = document.getElementById('video-container');

const videoWorks = !!document.createElement('video').canPlayType;
if (videoWorks) {
    video.controls = false;
    videoControls.classList.remove('hidden');
}

// Play/Pause BEGIN
const playButton = document.getElementById('play');
function togglePlay() {
    if (video.paused || video.ended) {
        video.play();
    }
    else {
        video.pause();
    }
}
playButton.addEventListener('click', togglePlay);

const playbackIcons = document.querySelectorAll('.playback-icons use');
function updatePlayButton() {
    playbackIcons.forEach(icon => icon.classList.toggle('hidden'));
    if (video.paused) {
        playButton.setAttribute('data-title', 'Play')
    }
    else {
        playButton.setAttribute('data-title', 'Pause')
    }
}
video.addEventListener('play', updatePlayButton);
video.addEventListener('pause', updatePlayButton);
// Play/Pause END

// Duration/Time elapsed BEGIN
const timeElapsed = document.getElementById('time-elapsed');
const duration = document.getElementById('duration');
function formatTime(timeInSeconds) {
    try {
        const result = new Date(timeInSeconds * 1000).toISOString().substr(11, 8);
        return {
            minutes: result.substr(3, 2),
            seconds: result.substr(6, 2),
        };
    }
    catch (e) {
        console.log('wrong time format');
        return {
            minutes: 'nan',
            seconds: 'nan',
        };
    }
};
// Duration/Time elapsed END

// Progress bar BEGIN
const progressBar = document.getElementById('progress-bar');
const seek = document.getElementById('seek');
function updateVideoInfo() {
    const videoDuration = Math.round(video.duration);
    console.log(videoDuration)
    seek.setAttribute('max', videoDuration);
    progressBar.setAttribute('max', videoDuration);
    const time = formatTime(videoDuration);
    console.log(time);
    duration.innerText = `${time.minutes}:${time.seconds}`;
    duration.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`);
    video.playbackRate = 0.88;
}
video.addEventListener('loadedmetadata', updateVideoInfo);
// Progress bar END

// Update function BEGIN
function updateTimeElapsed() {
    const time = formatTime(Math.round(video.currentTime));
    timeElapsed.innerText = `${time.minutes}:${time.seconds}`;
    timeElapsed.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`)
}

function updateProgress() {
    seek.value = Math.round(video.currentTime);
    progressBar.value = Math.round(video.currentTime);
}

function updateEverything() {
    updateVideoInfo();
    updateTimeElapsed();
    updateProgress();
}

setInterval(updateEverything, 500);
// Update function END

const seekTooltip = document.getElementById('seek-tooltip');
function updateSeekTooltip(event) {
    const skipTo = Math.round((event.offsetX / event.target.clientWidth) * parseInt(event.target.getAttribute('max'), 10));
    seek.setAttribute('data-seek', skipTo)
    const t = formatTime(skipTo);
    seekTooltip.textContent = `${t.minutes}:${t.seconds}`;
    const rect = video.getBoundingClientRect();
    seekTooltip.style.left = `${event.pageX - rect.left}px`;
}
seek.addEventListener('mousemove', updateSeekTooltip);

function skipAhead(event) {
    const skipTo = event.target.dataset.seek ? event.target.dataset.seek : event.target.value;
    video.currentTime = skipTo;
    progressBar.value = skipTo;
    seek.value = skipTo;
}
seek.addEventListener('input', skipAhead);
// UI END

var server_ip, flask_port;
server_ip = 'localhost';
flask_port = '16034';


// Click event START
function getClickCoordinate(event) {
    var x = event.clientX + window.scrollX - videoContainer.offsetLeft;
    var y = event.clientY + window.scrollY - videoContainer.offsetTop;
    x = x < 0 ? 0 : x;
    y = y < 0 ? 0 : y;

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open(
        'GET',
        `http://${server_ip}:${flask_port}/data?coor=` + x + ',' + y + ',' + video.offsetHeight + ',' + video.offsetWidth,
        true
    );
    xmlHttp.send(null);
    xmlHttp.onload = function () {
        var logBox = document.getElementById('log-container');
        var node = document.createElement('div');
        var today = new Date();
        node.innerHTML = `Select #${JSON.parse(xmlHttp.responseText).position} @ ${today.timeNow()}`;
        node.setAttribute('class', 'log-elem')
        logBox.insertBefore(node, logBox.firstChild);
    }

    event.preventDefault();
}
video.addEventListener('click', getClickCoordinate);
// Click event END

Date.prototype.timeNow = function () {
    return ((this.getHours() < 10) ? '0' : '') + this.getHours() + ':' + ((this.getMinutes() < 10) ? '0' : '') + this.getMinutes() + ':' + ((this.getSeconds() < 10) ? '0' : '') + this.getSeconds();
}