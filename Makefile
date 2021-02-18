all: movement_tracking

movement_tracking: movement_tracking.o
	g++ movement_tracking.o -o movement_tracking `pkg-config --libs opencv` 

movement_tracking.o: movement_tracking.cpp
	g++ -c movement_tracking.cpp `pkg-config --cflags opencv` 
	@echo 'Finished building target: $@'

clean:
	rm -rf *o movement_tracking
