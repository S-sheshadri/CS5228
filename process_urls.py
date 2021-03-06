"""
    The script downloads the urls in parallel, and it uses the newspaper library to extract the text from
    the htmls. IT doesn't do any further processing till now.
"""
import re
import threading
from newspaper import Article

num_threads = 4
non_alpha_regex = re.compile('[^a-zA-Z]')

def get_urls(urlpath):
    global num_threads
    urls = []
    with open(urlpath, "r") as fp:
        for line in fp:
           urls.append(line.strip().split("\t"))

    per_thread = int(len(urls)/num_threads)
    for thread in range(num_threads):
        yield urls[thread*per_thread: (thread + 1)*per_thread]

def worker(thread_num, 
            urls):
    url_text = []
    thread_file = open("test_data/Thread_"+str(thread_num) + ".dat", "a+")
    for (idx,url) in urls:
        print(thread_num, idx, url)
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
        except Exception as e:
            print("Error: " + str(e))
            thread_file.write("|".join([str(idx), "Empty"]))
            thread_file.write("##")
            continue
        
        thread_file.write("|".join([str(idx), text]))
        thread_file.write("##")
    thread_file.close()

threadn = 0
threads = []
for urls in get_urls("urls"):
    t = threading.Thread(target=worker, args=(threadn, urls))
    threads.append(t)
    t.start()
    print("Started Thread: ", str(threadn))
    threadn += 1
