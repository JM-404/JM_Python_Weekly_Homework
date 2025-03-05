n,m=map(int,input().split())
name=set(input().split())
for i in range(m):
    read=set(input().split())
    name=name-read
print(len(name))