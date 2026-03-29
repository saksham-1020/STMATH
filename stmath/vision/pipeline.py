from .vision import convolve2d, edge

class VisionPipeline:

    def run_edge_detection(self, img):
        return edge(img)