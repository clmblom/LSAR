import xml.etree.ElementTree as ET
from xml.dom import minidom


class PageWriter:
    def __init__(self):
        self.root = self.create_base()
        self.region_number = 0
        self.text_line_number = 0

    def create_base(self):
        ns_dict = {"xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                   "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                   "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"}
        root = ET.Element('PcGts', attrib=ns_dict)
        return root

    def add_page(self, image_name, image_width, image_height):
        image_dict = {"imageFilename": f"{image_name}", "imageWidth": f"{image_width}", "imageHeight": f"{image_height}"}
        return ET.SubElement(self.root, 'Page', attrib=image_dict)

    def add_text_region(self, page, coords):
        # coords is list of [(x_0, y_1), ..., (x_n, y_n)]
        self.region_number += 1
        tr = ET.SubElement(page, 'TextRegion', {'id': f"{str(self.region_number)}"})
        coord_str = self._coords_to_str(coords)
        ET.SubElement(tr, 'Coords', {'points': coord_str})
        self.text_line_number = 0
        return tr

    def add_base_line(self, text_line, coords):
        coord_str = self._coords_to_str(coords)
        return ET.SubElement(text_line, 'Baseline', {'points': coord_str})

    def add_text_line(self, text_region, coords, custom=None):
        if custom:
            tl = ET.SubElement(text_region, 'TextLine',
                               {'id': f"r{str(self.region_number)}l{str(self.text_line_number)}",
                                'custom': custom})
        else:
            tl = ET.SubElement(text_region, 'TextLine',
                               {'id': f"r{str(self.region_number)}l{str(self.text_line_number)}"})
        coord_str = self._coords_to_str(coords)
        ET.SubElement(tl, 'Coords', {'points': coord_str})
        self.text_line_number += 1
        return tl

    def add_text(self, text_line, text):
        te = ET.SubElement(text_line, 'TextEquiv')
        u = ET.SubElement(te, 'Unicode')
        u.text = text

    def write_xml(self, save_path):
        #et = ET.ElementTree(self.root)
        xml_string = ET.tostring(self.root, encoding='utf-8')
        xml_string = minidom.parseString(xml_string).toprettyxml(indent="    ", encoding='utf-8')
        with open(save_path, 'wb') as f:
            f.write(xml_string)

    def _coords_to_str(self, coords):
        coord_str = ' '.join([f"{str(int(x))},{str(int(y))}" for x, y in coords])
        return coord_str




