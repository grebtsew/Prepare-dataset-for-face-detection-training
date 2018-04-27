import xml.etree.cElementTree as ET
import os

class detected_object:
    def __init__(self):
        self.positions = [0,0,0,0]
        self.name = ""
        self.pose = "Unspecified"
        self.truncated = 0
        self.difficult = 0


#Create file for tensorflow learning

def create_training_xml(folder,
                        filename,
                        path,
                        database,
                        width,
                        height,
                        depth,
                        segmented,
                        detected_objects_array,
                        result_folder):
 
 #folder = "" str path to image file folder
 #filename = "" str image filename
 #path = "" str image full path
 #database = "" database "unknown"
 #width = "" image width
 #height = "" image height
 #depth = "" 
 #segmented = ""

 #detected_object_array = [] array of detected object struct ( contains all detected objects)   
 #name = "" object tag name
 #pose = "" 
 #truncated = "" 
 #difficult = "" 
 #xmin = ""  #ymin = "" #xmax = "" #ymax = "" box positions 
 #result_folder = "" Creates result xml i following folder with same name as image

 xml_annotation = ET.Element('annotation')

 ET.SubElement(xml_annotation, "folder").text = str(folder)
 ET.SubElement(xml_annotation, "filename").text = str(filename)
 ET.SubElement(xml_annotation, "path").text = str(path)
 xml_source = ET.SubElement(xml_annotation, "source")
 ET.SubElement(xml_source, "database").text = str(database)
 xml_size = ET.SubElement(xml_annotation, "size")
 ET.SubElement(xml_size, "width").text = str(width)
 ET.SubElement(xml_size, "height").text = str(height)
 ET.SubElement(xml_size, "depth").text = str(depth)
 ET.SubElement(xml_annotation, "segmented").text = str(segmented)

 for obj in detected_objects_array:
     xml_object = ET.SubElement(xml_annotation, "object")
     ET.SubElement(xml_object, "name").text = str(obj.name)
     ET.SubElement(xml_object, "pose").text = str(obj.pose)
     ET.SubElement(xml_object, "truncated").text = str(obj.truncated)
     ET.SubElement(xml_object, "difficult").text = str(obj.difficult)
     xml_bndbox = ET.SubElement(xml_object, "bndbox")
     ET.SubElement(xml_bndbox, "xmin").text = str(obj.positions[0])
     ET.SubElement(xml_bndbox, "ymin").text = str(obj.positions[1])
     ET.SubElement(xml_bndbox, "xmax").text = str(obj.positions[2])
     ET.SubElement(xml_bndbox, "ymax").text = str(obj.positions[3])
     
 file = ET.ElementTree(xml_annotation)
 file.write(''.join([result_folder,'/',  os.path.splitext(filename)[0],'.xml']))
 
 return
