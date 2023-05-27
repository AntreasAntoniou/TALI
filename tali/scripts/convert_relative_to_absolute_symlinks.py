find /tali-data/TALI/data -type l -exec bash -c 'f="{}"; target="$(readlink --canonicalize "$f")"; ln --force --symbolic "$target" "$f"' \;
