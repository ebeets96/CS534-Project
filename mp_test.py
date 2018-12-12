import multiprocessing as mp

def hello(other):
	print(other)
	for i in range(1000):
		a = 2
	print("Hi")

if __name__ == "__main__":
	p = mp.Pool(4)
	for i in range(10):
		res = p.apply_async(hello, args=(12,))
		res.get()

	print("before close")
	p.close()
	print("before join")
	p.join()
	print("joined")
