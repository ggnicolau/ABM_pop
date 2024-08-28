import os
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from scipy.stats import pareto, norm
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import gc
import random
from multiprocessing import Process

# Classe responsável por locação e violência no grid
class Locacao:
    def __init__(self, width, height, model):
        self.model = model
        self.matriz_valores = np.zeros((width, height))
        self.matriz_violencia = np.zeros((width, height))

    def atualizar_valores(self):
        for cell in self.model.grid.coord_iter():
            cell_content, pos = cell
            vizinhanca = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
            pessoas = [
                pessoa
                for neighbor in vizinhanca
                for pessoa in self.model.grid.get_cell_list_contents([neighbor])
                if isinstance(pessoa, AgenteUrbano)
            ]
            moradores = [pessoa for pessoa in pessoas if pessoa.obter_tipo() == "morador"]
            self.matriz_valores[pos[0], pos[1]] = len(moradores) / 9

    def obter_valor(self, x, y):
        return self.matriz_valores[x, y]

    def adicionar_violencia(self, x, y, quantidade):
        self.matriz_violencia[x, y] += quantidade

    def obter_violencia(self, x, y):
        return self.matriz_violencia[x, y]

# Agente Urbano, que pode ser do tipo "morador" ou "povo_rua"
class AgenteUrbano(Agent):
    def __init__(self, unique_id, model, tipo, limiar):
        super().__init__(unique_id, model)
        self.limiar = limiar
        self.tipo = tipo
        self.move_id = 0
        self.tolerancia_violencia = max(0, min(1, random.gauss(0.5, 0.1)))

    def mover(self):
        nova_posicao = (
            self.random.randrange(self.model.grid.width),
            self.random.randrange(self.model.grid.height),
        )
        self.model.grid.move_agent(self, nova_posicao)

    def obter_tipo(self):
        return self.tipo

    def step(self):
        valor_local = self.model.locacao.obter_valor(self.pos[0], self.pos[1])
        violencia_local = self.model.locacao.obter_violencia(self.pos[0], self.pos[1])

        if self.tipo == "povo_rua" and valor_local < self.limiar:
            self.mover()
            self.move_id = 1
        elif self.tipo == "morador" and (valor_local > self.limiar or violencia_local > self.tolerancia_violencia):
            self.mover()
            self.move_id = 1
        else:
            self.move_id = 0

# Agente Igreja, que se move conforme a concentração de agentes "povo_rua"
class AgenteIgreja(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.tipo = "igreja"
        self.concentracao_minima_povo_rua = 3

    def step(self):
        vizinhanca = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        vizinhos = self.model.grid.get_cell_list_contents(vizinhanca)
        quantidade_povo_rua = sum(1 for v in vizinhos if isinstance(v, AgenteUrbano) and v.obter_tipo() == "povo_rua")

        if quantidade_povo_rua < self.concentracao_minima_povo_rua:
            self.mover()

    def mover(self):
        passos_possiveis = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        melhor_posicao = None
        maior_concentracao = 0

        for passo in passos_possiveis:
            conteudo_celula = self.model.grid.get_cell_list_contents([passo])
            concentracao_povo_rua = sum(1 for agente in conteudo_celula if isinstance(agente, AgenteUrbano) and agente.obter_tipo() == "povo_rua")

            if concentracao_povo_rua > maior_concentracao:
                maior_concentracao = concentracao_povo_rua
                melhor_posicao = passo

        if melhor_posicao:
            self.model.grid.move_agent(self, melhor_posicao)
        else:
            nova_posicao = self.random.choice(passos_possiveis)
            self.model.grid.move_agent(self, nova_posicao)

# Agente Policia, que adiciona violência ao grid
class AgentePolicia(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.tipo = "policia"

    def step(self):
        self.model.locacao.adicionar_violencia(self.pos[0], self.pos[1], 0.2)
        self.mover()

    def mover(self):
        passos_possiveis = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        nova_posicao = self.random.choice(passos_possiveis)
        self.model.grid.move_agent(self, nova_posicao)

# Modelo base que organiza a simulação
class ModeloDinamicaUrbana(Model):
    def __init__(self, num_agentes, width, height, forma_morador, escala_morador, mu_povo_rua, sd_povo_rua):
        super().__init__()
        self.num_agentes = num_agentes
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        self.locacao = Locacao(width, height, self)

        for i in range(self.num_agentes):
            if i / self.num_agentes > 0.05:
                agente = AgenteUrbano(i, self, "morador", pareto.rvs(b=forma_morador) * escala_morador)
            else:
                agente = AgenteUrbano(i, self, "povo_rua", norm.rvs(mu_povo_rua, sd_povo_rua))
            self.schedule.add(agente)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agente, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "Mobilidade_Morador": self.calcular_mobilidade_morador,
                "Mobilidade_Povo_Rua": self.calcular_mobilidade_povo_rua,
                "Indice_Gini_Morador": self.calcular_indice_gini_morador,
                "Indice_Gini_Povo_Rua": self.calcular_indice_gini_povo_rua,
            }
        )

    def step(self):
        self.locacao.atualizar_valores()
        self.schedule.step()
        self.datacollector.collect(self)

    def calcular_mobilidade(self, tipo_agente):
        mobilidade_agente = [
            agente.move_id
            for agente in self.schedule.agents
            if isinstance(agente, AgenteUrbano) and agente.obter_tipo() == tipo_agente
        ]
        return sum(mobilidade_agente) / len(mobilidade_agente)

    def calcular_mobilidade_morador(self):
        return self.calcular_mobilidade("morador")

    def calcular_mobilidade_povo_rua(self):
        return self.calcular_mobilidade("povo_rua")

    def calcular_indice_gini(self, tipo_agente):
        ocupacao = np.zeros(self.grid.width * self.grid.height)
        for cell in self.grid.coord_iter():
            cell_content, pos = cell
            ocupacao[pos[0] * self.grid.height + pos[1]] = len(
                [agente for agente in cell_content if isinstance(agente, AgenteUrbano) and agente.obter_tipo() == tipo_agente]
            )
        ocupacao.sort()
        N = self.grid.height * self.grid.width
        B = sum(xi * (N - i) for i, xi in enumerate(ocupacao)) / (N * sum(ocupacao))
        return 1 + (1 / N) - 2 * B

    def calcular_indice_gini_morador(self):
        return self.calcular_indice_gini("morador")

    def calcular_indice_gini_povo_rua(self):
        return self.calcular_indice_gini("povo_rua")

# Modelo intermediário com igrejas
class ModeloComIgreja(ModeloDinamicaUrbana):
    def __init__(self, num_agentes, width, height, forma_morador, escala_morador, mu_povo_rua, sd_povo_rua, num_igrejas):
        super().__init__(num_agentes, width, height, forma_morador, escala_morador, mu_povo_rua, sd_povo_rua)
        for i in range(num_igrejas):
            igreja = AgenteIgreja(self.num_agentes + i, self)
            self.schedule.add(igreja)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(igreja, (x, y))

# Modelo completo com igrejas e polícia
class ModeloPoliciaIgreja(ModeloDinamicaUrbana):
    def __init__(self, num_agentes, width, height, forma_morador, escala_morador, mu_povo_rua, sd_povo_rua, num_igrejas, num_policias):
        super().__init__(num_agentes, width, height, forma_morador, escala_morador, mu_povo_rua, sd_povo_rua)
        for i in range(num_igrejas):
            igreja = AgenteIgreja(self.num_agentes + i, self)
            self.schedule.add(igreja)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(igreja, (x, y))

        for i in range(num_policias):
            policia = AgentePolicia(self.num_agentes + num_igrejas + i, self)
            self.schedule.add(policia)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(policia, (x, y))

# Funções de visualização e combinação de GIFs
class UtilidadesVisualizacao:
    @staticmethod
    def salvar_imagens_distribuicao_agentes(model, output_dir, tipo_agente, iteracoes):
        os.makedirs(output_dir, exist_ok=True)
        contagem_agentes = np.zeros((model.grid.width, model.grid.height))

        for passo in range(iteracoes):
            model.step()
            for cell in model.grid.coord_iter():
                cell_content, pos = cell
                agente_contagem = len(
                    [agente for agente in cell_content if isinstance(agente, AgenteUrbano) and agente.obter_tipo() == tipo_agente]
                )
                contagem_agentes[pos[0], pos[1]] = agente_contagem

            plt.figure(figsize=(8, 8))
            plt.imshow(contagem_agentes, interpolation="nearest")
            plt.colorbar()
            plt.title(f"Distribuição de {tipo_agente.capitalize()} - Passo {passo + 1}")

            temp_filename = os.path.join(output_dir, f"{tipo_agente}_passo_{passo+1:04d}.png")
            plt.savefig(temp_filename)
            plt.close()

    @staticmethod
    def combinar_imagens_para_gif(image_dir, output_gif):
        imagens = []
        for image_file in sorted(os.listdir(image_dir)):
            if image_file.endswith(".png"):
                imagens.append(imageio.imread(os.path.join(image_dir, image_file)))
        imageio.mimsave(output_gif, imagens, fps=2)

    @staticmethod
    def salvar_gif_distribuicao_combinada(model, output_dir, prefix, iteracoes):
        dir_morador = os.path.join(output_dir, f"{prefix}_morador")
        dir_povo_rua = os.path.join(output_dir, f"{prefix}_povo_rua")
        UtilidadesVisualizacao.salvar_imagens_distribuicao_agentes(model, dir_morador, "morador", iteracoes)
        UtilidadesVisualizacao.salvar_imagens_distribuicao_agentes(model, dir_povo_rua, "povo_rua", iteracoes)
        gif_morador = os.path.join(output_dir, f"{prefix}_morador.gif")
        gif_povo_rua = os.path.join(output_dir, f"{prefix}_povo_rua.gif")
        UtilidadesVisualizacao.combinar_imagens_para_gif(dir_morador, gif_morador)
        UtilidadesVisualizacao.combinar_imagens_para_gif(dir_povo_rua, gif_povo_rua)
        UtilidadesVisualizacao.combinar_gifs([gif_morador, gif_povo_rua], os.path.join(output_dir, f"{prefix}_combinado.gif"))

    @staticmethod
    def combinar_gifs(gif_files, output_file, titles=None, columns=2):
        gifs = [Image.open(gif) for gif in gif_files]
        num_frames = [gif.n_frames for gif in gifs]
        if len(set(num_frames)) > 1:
            raise ValueError("Todos os GIFs devem ter o mesmo número de frames para combinar corretamente.")

        width, height = gifs[0].size
        rows = (len(gif_files) + columns - 1) // columns
        total_width = columns * width
        total_height = rows * height + (rows * 50 if titles else 0)  # Ajuste de altura para títulos
        frames = []

        for frame_number in range(gifs[0].n_frames):
            new_frame = Image.new("RGB", (total_width, total_height), "white")
            draw = ImageDraw.Draw(new_frame)

            for idx, gif in enumerate(gifs):
                gif.seek(frame_number)
                x_offset = (idx % columns) * width
                y_offset = (idx // columns) * height + (50 if titles else 0)

                new_frame.paste(gif, (x_offset, y_offset))
                
                # Adiciona título acima da linha, se fornecido
                if titles:
                    title = titles[idx // columns]
                    draw.text((x_offset + width // 2, y_offset - 40), title, fill="black", anchor="mm", align="center")

            frames.append(new_frame)

        frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=100, loop=0)
        del frames
        gc.collect()

# Funções para rodar e salvar resultados dos modelos
class ExecutorModelo:
    @staticmethod
    def rodar_modelo_e_salvar_resultados(model_class, output_dir, prefix, iteracoes, **model_params):
        model = model_class(**model_params)
        
        # Cria subpastas para gifs e csvs
        gifs_dir = os.path.join(output_dir, "gifs")
        csvs_dir = os.path.join(output_dir, "csvs")
        os.makedirs(gifs_dir, exist_ok=True)
        os.makedirs(csvs_dir, exist_ok=True)

        UtilidadesVisualizacao.salvar_gif_distribuicao_combinada(model, gifs_dir, prefix, iteracoes)
        results = model.datacollector.get_model_vars_dataframe()
        results.to_csv(os.path.join(csvs_dir, f"{prefix}_resultados.csv"))
        del model
        gc.collect()

    @staticmethod
    def rodar_modelo_em_processo_separado(model_class, output_dir, prefix, iteracoes, **model_params):
        p = Process(target=ExecutorModelo.rodar_modelo_e_salvar_resultados, args=(model_class, output_dir, prefix, iteracoes), kwargs=model_params)
        p.start()
        p.join()

# Configuração e execução dos modelos
if __name__ == "__main__":
    num_agentes = 5000
    largura_grid = 10
    altura_grid = 10
    forma_morador = 4
    escala_morador = 35
    mu_povo_rua = 71
    sd_povo_rua = 57
    num_igrejas = 3
    num_policias = 2
    iteracoes = 200

    diretorio_output = "output"
    os.makedirs(diretorio_output, exist_ok=True)

    print("Executando o modelo básico...")
    ExecutorModelo.rodar_modelo_em_processo_separado(ModeloDinamicaUrbana, diretorio_output, "basico", iteracoes, num_agentes=num_agentes, width=largura_grid, height=altura_grid, forma_morador=forma_morador, escala_morador=escala_morador, mu_povo_rua=mu_povo_rua, sd_povo_rua=sd_povo_rua)

    print("Executando o modelo com Igreja...")
    ExecutorModelo.rodar_modelo_em_processo_separado(ModeloComIgreja, diretorio_output, "igreja", iteracoes, num_agentes=num_agentes, width=largura_grid, height=altura_grid, forma_morador=forma_morador, escala_morador=escala_morador, mu_povo_rua=mu_povo_rua, sd_povo_rua=sd_povo_rua, num_igrejas=num_igrejas)

    print("Executando o modelo com Polícia e Igreja...")
    ExecutorModelo.rodar_modelo_em_processo_separado(ModeloPoliciaIgreja, diretorio_output, "policia_igreja", iteracoes, num_agentes=num_agentes, width=largura_grid, height=altura_grid, forma_morador=forma_morador, escala_morador=escala_morador, mu_povo_rua=mu_povo_rua, sd_povo_rua=sd_povo_rua, num_igrejas=num_igrejas, num_policias=num_policias)

    # Combinar todos os GIFs em uma única imagem
    titles = ["Modelo Básico", "Modelo com Igreja", "Modelo com Polícia e Igreja"]
    UtilidadesVisualizacao.combinar_gifs([
        os.path.join(diretorio_output, "gifs", "basico_combinado.gif"),
        os.path.join(diretorio_output, "gifs", "igreja_combinado.gif"),
        os.path.join(diretorio_output, "gifs", "policia_igreja_combinado.gif")
    ], os.path.join(diretorio_output, "gifs", "todos_modelos_combinados.gif"), titles=titles, columns=1)
    gc.collect()

    # Carregar e comparar os resultados dos modelos
    resultados_basico = pd.read_csv(os.path.join(diretorio_output, "csvs", "basico_resultados.csv"), index_col=0).iloc[-1]
    resultados_igreja = pd.read_csv(os.path.join(diretorio_output, "csvs", "igreja_resultados.csv"), index_col=0).iloc[-1]
    resultados_completo = pd.read_csv(os.path.join(diretorio_output, "csvs", "policia_igreja_resultados.csv"), index_col=0).iloc[-1]

    df_comparacao = pd.DataFrame({
        "Basico": resultados_basico,
        "Igreja": resultados_igreja,
        "PoliciaIgreja": resultados_completo
    })

    print("\nComparação dos resultados dos modelos:")
    print(df_comparacao)

    df_comparacao.to_csv(os.path.join(diretorio_output, "csvs", "comparacao_modelos.csv"))

    print("\nTodos os GIFs foram gerados e combinados com sucesso. Os resultados foram comparados e salvos.")