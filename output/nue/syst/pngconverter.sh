for f in *.pdf; do
  name=$(echo "$f" | cut -f 1 -d '.')
  echo "$name"
  pdftoppm -png -rx 300 -ry 300 "$f" "./png/${name}"
done
