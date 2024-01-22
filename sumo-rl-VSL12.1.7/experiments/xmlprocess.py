


import glob
import xml.etree.ElementTree as ET
def change_xmlfile(path):

    tree = ET.parse(xml_file)
    obj_root=tree.getroot()
    obj_list = tree.getroot().findall('object')
    for obj in obj_list:
        if obj.find('name').text== 'suv':
            obj.find('name').text=new_name1
            i=i+1

        elif obj.find('name').text== 'license_plate':
            obj.find('name').text = new_name2
            i = i + 1
        elif obj.find('name').text== 'wheel':
            obj.find('name').text= new_name3
            i = i+ 1
        #删除label为'car'的标签
        elif obj.find('name').text== 'car':
            obj_root.remove(obj)

    tree.write(xml_file)    # 将改好的文件重新写入，会覆盖原文件
    print('共完成了{}处替换'.format(i))
    print('共完成了{}处删除'.format(j))
path = r'./Annotations'    # xml文件夹路径
change_xmlfile(path)
#ET.parse(file_path).getroot().remove()
#obj_root=ET.parse(file_path).getroot()
