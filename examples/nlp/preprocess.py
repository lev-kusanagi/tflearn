import unidecode

with open ("donquijote_spanish_only.txt", "r") as myfile:
    data=myfile.readlines()

new_data = []

for i, line in enumerate(data):
    new_data.append(unidecode.unidecode(line))

with open("donquijote.txt", "w") as myfile:
    for line in new_data:
        myfile.write("%s" % line)
