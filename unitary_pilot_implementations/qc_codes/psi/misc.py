def _coordinate_string(geom):
    """coordinate string
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :rtype: str
    """
    coord_line_template = "{:2s} {: >17.12f} {: >17.12f} {: >17.12f}"
    coord_str = "\n".join(coord_line_template.format(label, *coord)
                          for label, coord in geom)
    coord_str += "\nunits bohr\nnoreorient"
    return coord_str

