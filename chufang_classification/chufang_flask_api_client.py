import requests

# files = {'file': open('./data/chufang_data/negative/6.jpg', 'rb')}
# files = {'file': open('./data/chufang_data_processed/test/positive/8.jpg', 'rb')}
files = {'file': open('./uploaded_data/0.jpg', 'rb')}

r = requests.post("http://172.17.30.118:8080/check_chufang_file", files=files)
print('r.text is', r.text)






# files = {'file': open('./data/chufang_data/positive/1007.jpg', 'rb')}  # [0.0009495963, 0.99905044]
# files = {'file': open('./data/chufang_data_processed/val/positive/3.jpg', 'rb')}  # [0.002257867, 0.9977422]
# user_info = {'name': 'tsg'}

# r = requests.post("http://127.0.0.1:8080/check_chufang_image", data=user_info, files=files)