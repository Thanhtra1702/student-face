import os
import requests
from bs4 import BeautifulSoup
import sqlite3
import sys

# Đảm bảo in được tiếng Việt trên môi trường Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# HTML snippet provided by user
html_content = """
<tbody><tr><td>1</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/cab9f738-fa04-4799-9613-812c725e4b24.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190002</td><td>Thân</td><td>Phúc</td><td>Hậu</td></tr><tr><td>2</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/e9629483-38fa-4ec0-90ad-7e3c42ff2f72.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190012</td><td>Nguyễn</td><td>Châu Thành</td><td>Sơn</td></tr><tr><td>3</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/fd7a6ad8-79b4-4895-9891-5fdbb6443893.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190023</td><td>Phan</td><td>Minh</td><td>Tài</td></tr><tr><td>4</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/029c6ff0-b00b-4f97-b384-6cf1047dba63.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190027</td><td>Nguyễn</td><td>Hải</td><td>Nam</td></tr><tr><td>5</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/22bc40b3-0e12-4966-9a07-e13af0275e00.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190029</td><td>Võ</td><td>Quang</td><td>Trường</td></tr><tr><td>6</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/04965630-d831-475f-ac25-28392725b1f9.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190039</td><td>Tô</td><td>Thanh</td><td>Hậu</td></tr><tr><td>7</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/eac36c15-9ded-413b-9def-37dcd5a9a04f.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190047</td><td>Phạm</td><td>Quang</td><td>Chiến</td></tr><tr><td>8</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/dcb91148-f5cc-4618-8909-447ffffe7257.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190057</td><td>Nguyễn</td><td>Nhất</td><td>Long</td></tr><tr><td>9</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/76b7892c-e0ac-4d19-bdca-7fc91272b134.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190059</td><td>Võ</td><td>Minh</td><td>Huy</td></tr><tr><td>10</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/aaaf66b6-5b8b-42c0-a41d-98d18cee9beb.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190076</td><td>Ngô</td><td>Tuấn</td><td>Hoàng</td></tr><tr><td>11</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/91a6fffa-b3e2-4616-9e0b-c51284ee0abe.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190099</td><td>Nguyễn</td><td>Thanh</td><td>Trà</td></tr><tr><td>12</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/50ebd337-c780-4b61-b75f-a315ffc2fecd.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190109</td><td>Châu</td><td>Thái Nhật</td><td>Minh</td></tr><tr><td>13</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/0dd240c6-119f-4ed8-9d28-2cdc1f915a54.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190123</td><td>Phan</td><td>Đỗ Thanh</td><td>Tuấn</td></tr><tr><td>14</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/4b1dbafb-6b52-4f17-9b1e-68656d8e686e.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190129</td><td>Trịnh</td><td>Khải</td><td>Nguyên</td></tr><tr><td>15</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/c13bf825-2d34-46c0-a65a-ca6000848609.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190134</td><td>Nguyễn</td><td>Lê Anh</td><td>Duy</td></tr><tr><td>16</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/8bb12a4e-b1da-499e-85f5-f20ad64c109f.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190136</td><td>Đặng</td><td>Văn</td><td>Hậu</td></tr><tr><td>17</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/9bb0b6b6-0f53-442e-ad6e-b8f78623817e.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190142</td><td>Lê</td><td>Hoàng</td><td>Hữu</td></tr><tr><td>18</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/62dd29b1-0cac-41ee-b26d-d1468a06dabe.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190155</td><td>Nguyễn</td><td>Lê Tấn</td><td>Pháp</td></tr><tr><td>19</td><td><center><img src="https://fap.fpt.edu.vn/temp/ImageRollNumber/QN/b4b926b4-2cdd-489c-b35a-f2059d224e9f.jpg" style="height:146px;width:111px;border-width:0px;"></center></td><td>QE190162</td><td>Trần</td><td>Gia</td><td>Phúc</td></tr></tbody>
"""

def extract_and_download():
    soup = BeautifulSoup(html_content, 'html.parser')
    rows = soup.find_all('tr')
    
    database_dir = 'database'
    if not os.path.exists(database_dir):
        os.makedirs(database_dir)
        
    student_data = []
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 6:
            img_url = cols[1].find('img')['src']
            mssv = cols[2].text.strip()
            # FAP name format: Surname, Middle Name, First Name
            full_name = f"{cols[3].text.strip()} {cols[4].text.strip()} {cols[5].text.strip()}"
            
            print(f"Processing {mssv} - {full_name}...")
            
            # Download image
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                    'Referer': 'https://fap.fpt.edu.vn/'
                }
                response = requests.get(img_url, headers=headers, stream=True)
                if response.status_code == 200:
                    img_path = os.path.join(database_dir, f"{mssv}.jpg")
                    with open(img_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"  Downloaded image to {img_path}")
                else:
                    print(f"  Failed to download image for {mssv}")
            except Exception as e:
                print(f"  Error downloading {mssv}: {e}")
                
            student_data.append((mssv, full_name))
            
    return student_data

if __name__ == "__main__":
    data = extract_and_download()
    print("\nExtraction complete. Formatting for setup_database.py...")
    for mssv, name in data:
        # Schedule and room as placeholders
        print(f"('{mssv}', '{name}',")
