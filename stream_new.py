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
    fps = 10.0      # 接收與輸出的 fps
    speed_up = 1.2    # 加速兩倍
    clean_and_mkdir('hls')
    (
        ffmpeg
        .input(server_url, r=fps)
        .setpts(f'{1/speed_up}*PTS')        
        .output(
            'hls/hls.m3u8',
            format='hls',
            vcodec='libx264',
            pix_fmt='yuv420p',
            preset='veryfast',
            g='10',
            r=str(fps*speed_up),
            fflags="nobuffer",
            flags="low_delay"
        )
        .global_args("-re")
        .run()
    )