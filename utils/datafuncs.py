import xml.etree.ElementTree as ET


def alto_parser_rec(parent, line_list):
    if parent.tag.split('}')[-1] == 'TextLine':
        baseline_coords = [tuple([int(coord) for coord in points.split(',')]) for points in
                           parent.attrib['BASELINE'].split()]
        string_line_dict = {'baseline': baseline_coords}
        for string_line in parent:
            for k in string_line.attrib:
                if k not in {'ID', 'CONTENT'}:
                    string_line_dict[k.lower()] = int(string_line.attrib[k])
                elif k == 'CONTENT':
                    string_line_dict['text'] = string_line.attrib[k]
        line_list.append(string_line_dict)
    elif len(parent):
        for child in parent:
            alto_parser_rec(child, line_list)


def alto_parser(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    line_list = list()
    alto_parser_rec(root, line_list)
    return line_list


def page_parser_rec(parent, line_list, search_term):
    if parent.tag.split('}')[-1] == search_term:
        line_list.append(parent)
    else:
        for child in parent:
            page_parser_rec(child, line_list, search_term)


def page_parser(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    line_list = list()
    line_dict_list = list()
    page_parser_rec(root, line_list, 'TextLine')
    for line in line_list:
        d = dict()
        for child in line:
            if child.tag.split('}')[-1] == 'Coords':
                d['poly_coords'] = child.attrib['points']
            if child.tag.split('}')[-1] == 'Baseline':
                d['baseline_coords'] = child.attrib['points']
        line_dict_list.append(d)
    page_list = list()
    page_parser_rec(root, page_list, 'Page')
    return line_dict_list, page_list[0].attrib
