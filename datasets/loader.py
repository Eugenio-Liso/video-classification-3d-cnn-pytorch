from PIL import Image


class ImageLoader(object):

    def __init__(self):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            self.loader = self.__accimage_loader
        else:
            self.loader = self.__pil_loader

    def __call__(self, path):
        return self.loader(path)

    def __pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __accimage_loader(self, path):
        import accimage
        try:
            return accimage.Image(str(path))
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.__pil_loader(path)


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoader()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video