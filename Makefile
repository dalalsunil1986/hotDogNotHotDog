# By default, build the `run` target
all: run

# Rule to download zipfile
videoFrames.zip:
	curl -L 'https://www.dropbox.com/s/yvkdq5kiait86b9/videoFrames.zip?dl=1' -o $@

# Rule to extract the video frames; touch it to set the proper timestamp
videoFrames: videoFrames.zip
	unzip $< -x "__MACOSX/*"
	touch $@

# From here on out, we'll run python commands within a docker container
DOCKERIZE = docker run -v $$(pwd):/app -w /app -ti jwfromm/mxa_cpu

# Rule to generate the .lst files
mk_train.lst: videoFrames
	$(DOCKERIZE) python3 im2rec.py --list --recursive --num-thread 4 --train-ratio 0.6 --test-ratio 0.2 mk videoFrames

# Rule to generate the .rec files
mk_train.rec: mk_train.lst
	$(DOCKERIZE) python3 im2rec.py --num-thread 4 mk videoFrames

# Rule to pop open an interactive IPython shell within the docker container
shell:
	$(DOCKERIZE) ipython3

# Rule to run the whole shebang and generate some params!
run: mk_train.rec
	$(DOCKERIZE) python3 notHotDog.py

# Rule to cleanup after ourselves.  Purposefully does not delete params.
clean:
	rm -f *.idx *.rec *.lst *.zip
	rm -rf videoFrames
