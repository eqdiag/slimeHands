f = open("dummy.txt", "r")

lines = f.readlines()
for line in lines:
    spits = line.splitlines("\t")
    print(spits)