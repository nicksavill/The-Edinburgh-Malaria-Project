# make sure static/ is in Vaccines/ directory
if [ ! -d Vaccines/static ]; then
    echo "Error: static/ directory not found in Vaccines/"
    exit 1
fi

# compile the C code if it has changed
if [ ! -f Vaccines/Edinburgh_Model/timestep.so ] || [ Vaccines/Edinburgh_Model/timestep.so -ot Vaccines/Edinburgh_Model/timestep.c ]; then
    echo "Compiling C code..."
    cc -fPIC -O3 -shared -o Vaccines/Edinburgh_Model/timestep.so Vaccines/Edinburgh_Model/timestep.c
else
    echo "C code is up to date."
fi

# kill any existing bokeh server processes
pkill -f "bokeh"

# start the bokeh server from temp-webroot
nohup bokeh serve Vaccines/ --port 5006 --allow-websocket-origin=temp.bio.ed.ac.uk

# bokeh serve Vaccines/ --port=5006