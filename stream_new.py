import ffmpeg
import os

video_format = "flv"
server_url = "http://localhost:16034/video_feed"

def clean_and_mkdir(dirname):
    os.makedirs(dirname, exist_ok=True)
    files = [f for f in os.listdir(dirname)]
    for f in files:
        os.remove(os.path.join(dirname, f))


if __name__ == '__main__':
    clean_and_mkdir('hls')
    (
        ffmpeg
        .input(server_url, r=7.0)
        .output(
            'hls/hls.m3u8',
            format='hls',
            vcodec='libx264',
            pix_fmt='yuv420p',
            preset='veryfast',
            g='10',
            r=7.0,
            fflags="nobuffer",
            flags="low_delay"
        )
        .global_args("-re") # argument to act as a live stream
        .run()
    )