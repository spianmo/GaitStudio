import imageio
from slicerator import Slicerator


@Slicerator.from_class
class Video(object):
    """
    A class to simplify the per frame access in video files

    This class uses imageio and ffmpeg to handle video files
    and slicerator to allow for numpy array like access to
    the video frames (fancy-slicable iterable)

    Attributes
    ----------
    metadata : dict
        all video metadata given by imageio as a dict

    duration : float
        direct access to the 'duration' meta data (video duration)

    fps : float
        direct access to the 'fps' meta data (video framerate)

    size : tuple
        direct access to the 'size' meta data (width, height)

    Methods
    -------
    __getitem__(slice) : Iterable
        return the video frames given by the numpy style slicing
    """

    propagate_attrs = ['fps', 'duration', 'size', 'metadata']

    def __init__(self, video):
        """
        Parameters
        ----------
        video : Video
            file-like path to the used video
        """

        self._video = video
        self._reader = imageio.get_reader(video, 'ffmpeg')
        self._meta = self._reader.get_meta_data()
        self._nframes = int(self._meta['duration'] * self._meta['fps'])

    ## convenience accessors to metadata
    @property
    def metadata(self):
        return self._meta

    @property
    def duration(self):
        return self._meta['duration']

    @property
    def fps(self):
        return self._meta['fps']

    @property
    def size(self):
        w, h = self._meta['size']  # is this w,h or h,w ??
        return w, h

    def close(self):
        self._reader.close()

    ## Methods for slicerator
    def __getitem__(self, i):
        return self._reader.get_data(i)

    def __len__(self):
        return self._nframes

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
