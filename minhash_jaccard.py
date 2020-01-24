with open ('token.csv') as f:
    s1 = set(f.read())

with open('sad.csv') as s:
    s2 = set(s.read())

actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for dataset and sad text is", actual_jaccard)

import matplotlib.pyplot as plt
d1 = 'Depressive', 'Non-depressive'
d2 = [56.51,43.49]
colors = ['blue', 'yellow']
figureObject, axesObject = plt.subplots()
axesObject.pie(d2,



               autopct='%1.2f',

               startangle=90)

# Aspect ratio - equal means pie is a circle
patches, texts = plt.pie(d2, colors=colors, shadow=False, startangle=90)
plt.legend(patches, d1, loc="best")
axesObject.axis('equal')

plt.show()

from datasketch import MinHash
m1, m2 = MinHash(), MinHash()
with open ('token.csv') as f:
    tokens = f.read()

    for line in tokens:
        #file_docs.append(line)
        m1.update(line.encode('utf8'))

with open('sad.csv') as s:
    sad = s.read();
    for d in sad:
        m2.update(d.encode('utf8'))

print("Estimated Jaccard for dataset and sad text is", m1.jaccard(m2))



import matplotlib.pyplot as plt
d1 = 'Depressive', 'Non-depressive'
d2 = [55.47,44.53]
colors = ['green', 'red']
figureObject, axesObject = plt.subplots()
axesObject.pie(d2,



               autopct='%1.2f',

               startangle=90)

# Aspect ratio - equal means pie is a circle
patches, texts = plt.pie(d2, colors=colors, shadow=False, startangle=90)
plt.legend(patches, d1, loc="best")
axesObject.axis('equal')

plt.show()


