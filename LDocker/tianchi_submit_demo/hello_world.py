#
import json

s_num, l_num = 0, []
with open("/tcdata/num_list.csv", "r") as f:
    for i in f:
        i = i.strip("\n")
        if i:
            i = int(i)
            s_num += i
            l_num.append(i)
l_num.sort(reverse=True)
print("000", s_num, l_num[:10])

result = {
    "Q1": "Hello world",
    "Q2": s_num,
    "Q3": l_num[:10]
}
with open("result.json", "w") as f:
    json.dump(result, f)

print("001", "Hello world")



