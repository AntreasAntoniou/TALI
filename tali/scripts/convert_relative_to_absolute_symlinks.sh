find /tali-data/TALI/data -type l -exec bash -c 'f="{}"; target="$(readlink --canonicalize "$f")"; ln --force --symbolic "$target" "$f"' \;
7z x video_data_part.7z.001
apt install p7zip-full