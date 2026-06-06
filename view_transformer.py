import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        # Coordenadas em píxeis (o trapézio no vídeo)
        self.pixel_vertices = np.array([
            [110, 1035], 
            [265, 275], 
            [910, 260], 
            [1900, 915] 
        ]).astype(np.float32)
        
        # Coordenadas reais em metros (o retângulo 2D perfeito)
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ]).astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        
        # Verifica se o jogador está dentro do nosso novo trapézio esticado
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted', None)
                    if position is not None:
                        position_transformed = self.transform_point(np.array(position))
                        if position_transformed is not None:
                            position_transformed = position_transformed.squeeze().tolist()
                        tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
