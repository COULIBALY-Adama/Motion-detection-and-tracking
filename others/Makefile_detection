all: detection_mouvement

detection_mouvement: detection_mouvement.o
	g++ detection_mouvement.o -o detection_mouvement `pkg-config --libs opencv` 

detection_mouvement.o: detection_mouvement.cpp
	g++ -c detection_mouvement.cpp `pkg-config --cflags opencv` 
	@echo 'Finished building target: $@'

clean:
	rm -rf *o detection_mouvement
