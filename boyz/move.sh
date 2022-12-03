

cd images

echo ls -a

# make ffmpeg out of all images
ffmpeg -framerate 30 -i %05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ../out.mp4