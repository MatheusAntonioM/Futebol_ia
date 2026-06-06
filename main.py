from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from tactical_analyst import TacticalAnalyst 
import numpy as np

def main():
    print("1. Lendo o vídeo...")
    video_frames = read_video('input_videos/video_futebol.mp4')
    
    print("2. Iniciando Tracker...")
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    print("3. Interpolando a bola e adicionando posições base...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracker.add_position_to_tracks(tracks)
    
    print("4. Estabilizando a Câmara (Optical Flow)...")
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement = camera_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_stub.pkl')
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    print("5. Transformando Perspetiva (De Píxeis para Metros)...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    print("6. Calculando Velocidade e Distância (Km/h)...")
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    print("7. Separando Equipas e Posse de Bola...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            cor = team_assigner.team_colors[team]
            tracks['players'][frame_num][player_id]['team_color'] = (int(cor[0]), int(cor[1]), int(cor[2]))

    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

    print("8. Desenhando Gráficos no Vídeo...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    output_video_frames = speed_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    print("9. Guardando o ficheiro final...")
    save_video(output_video_frames, 'output_videos/video_saida.mp4')
    print("Sistema de análise visual concluído!")
    print("\n10. A extrair telemetria para o Analista Tático (Gemma)...")
    
    posse_equipa_1 = 0
    posse_equipa_2 = 0
    velocidade_maxima = 0
    distancia_maxima = 0

    # Percorre os dados gravados para extrair as métricas globais
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Posse de Bola
            if track.get('has_ball', False):
                if track.get('team') == 1:
                    posse_equipa_1 += 1
                elif track.get('team') == 2:
                    posse_equipa_2 += 1
            
            # Físico (Pico de velocidade e Distância máxima)
            if track.get('speed', 0) > velocidade_maxima:
                velocidade_maxima = track['speed']
            
            if track.get('distance', 0) > distancia_maxima:
                distancia_maxima = track['distance']

    # Cálculo das percentagens
    total_posse_frames = posse_equipa_1 + posse_equipa_2
    perc_eq1 = (posse_equipa_1 / total_posse_frames * 100) if total_posse_frames > 0 else 0
    perc_eq2 = (posse_equipa_2 / total_posse_frames * 100) if total_posse_frames > 0 else 0

    # Cria o dicionário de estatísticas para enviar ao Gemma
    match_stats = {
        "Posse de Bola - Equipa 1": f"{perc_eq1:.1f}%",
        "Posse de Bola - Equipa 2": f"{perc_eq2:.1f}%",
        "Pico de Velocidade Máxima Registado": f"{velocidade_maxima:.2f} km/h",
        "Distância Máxima Percorrida num Sprint": f"{distancia_maxima:.2f} metros",
    }

    # Inicia o Analista (Certifique-se que o ficheiro tactical_analyst.py está na mesma pasta)
    analyst = TacticalAnalyst(model_name="gemma4:latest") 
    relatorio = analyst.analyze(match_stats)

    print("\n==================================================")
    print(" RELATÓRIO TÁTICO DO JOGO ")
    print("==================================================")
    print(relatorio)
    print("==================================================")

if __name__ == '__main__':
    main()
