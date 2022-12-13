from arenets.features.utils import create_filled_array


class FrameConnotationFeatures(object):

    @ staticmethod
    def to_input(frame_inds, frame_sent_roles, size, filler):
        assert(isinstance(frame_inds, list))
        assert(isinstance(frame_sent_roles, list))
        assert(len(frame_inds) == len(frame_sent_roles))

        vector = create_filled_array(size=size, value=filler)

        for frame_ind, frame_ind_in_sample in enumerate(frame_inds):
            if frame_ind_in_sample >= len(vector):
                continue
            vector[frame_ind_in_sample] = frame_sent_roles[frame_ind]

        return vector
