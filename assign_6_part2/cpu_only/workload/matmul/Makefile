all: all-gem5 all-native
all-gem5: mm-ijk-gem5 mm-ijk-gem5-asm mm-ikj-gem5 mm-ikj-gem5-asm
all-native: mm-ijk-native mm-ikj-native

CC=riscv64-unknown-linux-gnu-g++

clean:
	rm mm-ijk-gem5 mm-ijk-gem5-asm mm-ikj-gem5 mm-ikj-gem5-asm
	rm mm-ijk-native mm-ikj-native

mm-ijk-native: mm.cpp
	g++ mm.cpp -o mm-ijk-native -static -O3 -DIJK

mm-ijk-gem5: mm.cpp
	$(CC) mm.cpp -o mm-ijk-gem5 -static -O3 -DIJK -I../include -DGEM5 -L../lib/riscv -lm5

mm-ijk-gem5-asm: mm.cpp
	$(CC) mm.cpp -o mm-ijk-gem5-asm -static -O3 -DIJK -I../include -DGEM5 -L../lib/riscv -lm5 -S -fverbose-asm

mm-ikj-native: mm.cpp
	g++ mm.cpp -o mm-ikj-native -static -O3 -DIKJ

mm-ikj-gem5: mm.cpp
	$(CC) mm.cpp -o mm-ikj-gem5 -static -O3 -DIKJ -I../include -DGEM5 -L../lib/riscv -lm5

mm-ikj-gem5-asm: mm.cpp
	$(CC) mm.cpp -o mm-ikj-gem5-asm -static -O3 -DIKJ -I../include -DGEM5 -L../lib/riscv -lm5 -S -fverbose-asm
