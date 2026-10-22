import pygame
import numpy as np
import torch
import random
import math
from collections import deque
from env.maze_env import MazeEnv
from agent.dqn import DQNAgent

CELL = 44  # Légèrement agrandi pour plus de détails
FPS  = 6   # Un poil plus rapide pour une transition fluide
SEQ_LEN = 8

def draw_grass_wall(screen, x, y, cell):
    """Dessine un mur de végétation dense avec ombrage et relief (Inspiré de image_4f0194.png)"""
    # 1. Ombre portée (sur le sol, décalée vers le bas et la droite)
    shadow_rect = pygame.Rect(x + 4, y + 4, cell, cell)
    pygame.draw.rect(screen, (220, 220, 225), shadow_rect, border_radius=4)

    # 2. Base du bloc de haie
    wall_rect = pygame.Rect(x, y, cell, cell)
    pygame.draw.rect(screen, (20, 80, 20), wall_rect, border_radius=5)

    # 3. Génération de la texture organique fixe
    random.seed(int(x * 777 + y * 999))
    
    # Dessin de mini-touffes de feuilles imbriquées
    for _ in range(120):
        # Rayon concentré vers le centre pour un effet de buisson bombé
        radius = random.randint(3, 6)
        cx = random.randint(4, cell - 4)
        cy = random.randint(4, cell - 4)
        
        # Palette de verts inspirée directement de image_4f0194.png
        color = random.choice([
            (34, 139, 34),    # Vert herbe standard
            (46, 175, 46),    # Vert clair lumineux
            (15, 70, 15),     # Ombre interne
            (110, 220, 70),   # Pointe de feuille au soleil
        ])
        
        # Dessin de la feuille (ellipse ou cercle)
        pygame.draw.circle(screen, color, (x + cx, y + cy), radius)

    # 4. Contour de finition pour la structure
    pygame.draw.rect(screen, (10, 50, 10), wall_rect, width=1, border_radius=5)


def draw_agent(screen, cx, cy):
    """Dessine un agent explorateur épuré et moderne (Style icône vectorielle)"""
    # Ombre de l'agent
    pygame.draw.circle(screen, (210, 215, 210), (cx + 2, cy + 3), 10)
    
    # Corps principal (Blanc / Chrome techno)
    pygame.draw.circle(screen, (250, 250, 255), (cx, cy), 10)
    pygame.draw.circle(screen, (60, 120, 240), (cx, cy), 10, width=2)  # Contour bleu
    
    # Visière luminescente (Donne une direction / un style IA)
    visiere_rect = pygame.Rect(cx - 5, cy - 4, 10, 5)
    pygame.draw.rect(screen, (0, 200, 255), visiere_rect, border_radius=2)
    
    # Petit noyau central
    pygame.draw.circle(screen, (60, 120, 240), (cx, cy + 4), 2)


def draw_goal(screen, x, y, cell, ticks):
    """Dessine un portail de sortie doré et animé"""
    cx = x * cell + cell // 2
    cy = y * cell + cell // 2
    
    # Effet d'aura pulsante
    pulse = int(4 * math.sin(ticks * 0.2))
    pygame.draw.circle(screen, (255, 230, 150), (cx, cy), 14 + pulse, width=1)
    
    # Le Coeur d'or
    pygame.draw.circle(screen, (255, 180, 0), (cx, cy), 10)
    pygame.draw.circle(screen, (255, 220, 50), (cx, cy), 7)
    pygame.draw.circle(screen, (255, 255, 180), (cx, cy), 3)


def run_visual(model_path="model.pth", maze_size=9):
    # Désactivation du rendu natif obsolète pour laisser place au nouveau design
    env = MazeEnv(size=maze_size, render_mode=None)
    
    agent = DQNAgent(seq_len=SEQ_LEN)
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.q_net.eval()

    pygame.init()
    
    W = maze_size * CELL
    screen = pygame.display.set_mode((W, W + 80))  # HUD légèrement plus grand
    pygame.display.set_caption("🌿 Intelligent Maze Agent")
    clock = pygame.time.Clock()
    
    # Polices d'écriture modernes (Sans-Serif et nettes)
    font_ui = pygame.font.SysFont("Helvetica, Arial, sans-serif", 14, bold=True)
    font_val = pygame.font.SysFont("Courier New, monospace", 16, bold=True)

    episode = 0
    ticks = 0
    
    while True:
        obs, _ = env.reset()
        obs_seq = deque(
            [np.zeros(9, dtype=np.float32)] * (SEQ_LEN - 1) + [obs.astype(np.float32)],
            maxlen=SEQ_LEN
        )
        trail = list()  # Utilisation d'une liste pour faire un effet de fondu ordonné
        done = False
        steps = 0

        while not done:
            ticks += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return

            # Fond d'écran : Gris-bleu très doux et lumineux (Premium feel)
            screen.fill((240, 242, 245))

            # Dessin du sol et des pistes
            for r in range(maze_size):
                for c in range(maze_size):
                    x = c * CELL
                    y = r * CELL
                    
                    if env.maze[r, c] == 1:
                        draw_grass_wall(screen, x, y, CELL)
                    else:
                        # Grille de fond subtile
                        pygame.draw.rect(screen, (248, 249, 250), (x, y, CELL, CELL))
                        pygame.draw.rect(screen, (230, 232, 235), (x, y, CELL, CELL), width=1)
                        
                        # Dessin des traces de pas de l'agent (Effet dégradé)
                        if (r, c) in trail:
                            idx = trail.index((r, c))
                            # Plus la trace est ancienne, plus elle est transparente/claire
                            alpha_factor = min(255, 130 + (idx * 5))
                            pygame.draw.circle(screen, (210, 230, 255), (x + CELL//2, y + CELL//2), 5)

            # Sortie Stylisée
            gr, gc = env.goal_pos
            draw_goal(screen, gc, gr, CELL, ticks)

            # Dessin de l'Agent
            ar, ac = env.agent_pos
            if (ar, ac) not in trail:
                trail.append((ar, ac))
            cx = ac * CELL + CELL // 2
            cy = ar * CELL + CELL // 2
            draw_agent(screen, cx, cy)

            # --- DESIGN DU HUD (Bas d'écran style Dashboard) ---
            hud_y = maze_size * CELL
            # Fond sombre mat
            pygame.draw.rect(screen, (24, 28, 36), (0, hud_y, W, 80))
            # Ligne fine de séparation technologique (Bleu électrique)
            pygame.draw.rect(screen, (60, 130, 240), (0, hud_y, W, 2))

            # Textes "Episode"
            ep_label = font_ui.render("EPISODE", True, (110, 120, 140))
            ep_text  = font_val.render(f"{episode:03d}", True, (255, 255, 255))
            screen.blit(ep_label, (20, hud_y + 18))
            screen.blit(ep_text, (20, hud_y + 38))

            # Textes "Steps"
            st_label = font_ui.render("TOTAL STEPS", True, (110, 120, 140))
            st_text  = font_val.render(f"{steps:04d}", True, (0, 210, 255))
            screen.blit(st_label, (160, hud_y + 18))
            screen.blit(st_text, (160, hud_y + 38))
            
            # Badge de statut "LIVE AI"
            pygame.draw.rect(screen, (40, 160, 80), (W - 95, hud_y + 28, 75, 22), border_radius=4)
            status_text = font_ui.render("AI LIVE", True, (255, 255, 255))
            screen.blit(status_text, (W - 83, hud_y + 32))

            pygame.display.flip()
            clock.tick(FPS)

            # Calcul IA
            with torch.no_grad():
                t = torch.tensor(np.array(obs_seq), dtype=torch.float32).unsqueeze(0)
                q_vals, _ = agent.q_net(t)
                action = q_vals.argmax().item()

            obs, _, terminated, truncated, _ = env.step(action)
            obs_seq.append(obs.astype(np.float32))
            done = terminated or truncated
            steps += 1

        episode += 1

if __name__ == "__main__":
    run_visual()