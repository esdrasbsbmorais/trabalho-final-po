#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Otimizacao Automatica - Deteccao de Tipos
Autor: Sistema de Otimizacao Inteligente
"""

import subprocess
import sys
from typing import List, Union, Tuple


class PatternSearch:
    """Implementacao do algoritmo Pattern Search com deteccao automatica de tipos"""

    def __init__(self, comando_modelo: str, modo: str = 'maximizar'):
        """
        Inicializa o Pattern Search

        Args:
            comando_modelo: Comando do executavel (ex: 'modelo10.exe')
            modo: 'maximizar' ou 'minimizar' (padrao: 'maximizar')
        """
        self.comando_modelo = comando_modelo
        self.modo = modo.lower()
        self.melhor_valor = float('-inf') if self.modo == 'maximizar' else float('inf')
        self.melhor_params = None
        self.historico = []
        self.total_execucoes = 0

    def detectar_padrao(self, padrao: str) -> Tuple[List, List[str]]:
        """
        Detecta automaticamente o padrao e tipos das variaveis

        Args:
            padrao: String com valores de exemplo (ex: "baixo 0 0.0 TRUE 1")

        Returns:
            Tupla com (valores_iniciais, tipos_detectados)
        """
        valores_str = padrao.strip().split()
        valores = []
        tipos = []

        for val in valores_str:
            # 1. Tenta INT primeiro
            try:
                valor_int = int(val)
                # Se for 0, transforma em 1 (nao pode comecar com 0)
                if valor_int == 0:
                    valor_int = 1
                valores.append(valor_int)
                tipos.append('int')
                continue
            except ValueError:
                pass

            # 2. Tenta FLOAT
            try:
                valor_float = float(val)
                # Se for 0.0, transforma em 1.0 (nao pode comecar com 0.0)
                if valor_float == 0.0:
                    valor_float = 1.0
                valores.append(valor_float)
                tipos.append('float')
                continue
            except ValueError:
                pass

            # 3. E TEXTUAL (strings como 'baixo', 'medio', 'alto', 'TRUE', etc)
            # Mantem o valor original em minusculo
            valores.append(val.lower())
            tipos.append('str')

        return valores, tipos

    def executar_modelo(self, params: List) -> float:
        """Executa o modelo com os parametros dados"""
        try:
            # Converte parametros para string
            params_str = [str(p) for p in params]

            # Executa o comando
            resultado = subprocess.run(
                [self.comando_modelo] + params_str,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Pega o valor de saida
            saida = resultado.stdout.strip()

            # Tenta extrair o numero da saida (pode vir como "Valor de saída: 123.45")
            if ":" in saida:
                # Extrai a parte apos os dois pontos
                valor = float(saida.split(":")[-1].strip())
            else:
                # Saida direta (apenas o numero)
                valor = float(saida)

            self.total_execucoes += 1

            # Atualiza melhor resultado
            melhorou = False
            if self.modo == 'maximizar':
                if valor > self.melhor_valor:
                    melhorou = True
            else:  # minimizar
                if valor < self.melhor_valor:
                    melhorou = True

            if melhorou:
                self.melhor_valor = valor
                self.melhor_params = params.copy()
                tipo_otimo = "maximo" if self.modo == 'maximizar' else "minimo"
                print(f"[OK] Novo {tipo_otimo}: {valor:.6f} com {params}")

            # Salva no historico
            self.historico.append((params.copy(), valor))

            return valor

        except ValueError as e:
            # Erro ao converter para float - provavelmente o modelo retornou uma mensagem de erro
            print(f"[ERRO] Modelo retornou mensagem de erro com parametros {params}:")
            print(f"       '{resultado.stdout.strip()}'")
            return float('-inf')
        except Exception as e:
            print(f"[ERRO] Erro ao executar modelo com parametros {params}: {e}")
            return float('-inf')

    def gerar_vizinhos(self, params: List, tipos: List[str], step_size: Union[int, float], limite_min: float, limite_max: float) -> List[List]:
        """
        Gera vizinhos para cada parametro baseado no tipo

        Args:
            params: Parametros atuais
            tipos: Tipos de cada parametro
            step_size: Tamanho do passo
            limite_min: Limite minimo
            limite_max: Limite maximo

        Returns:
            Lista de vizinhos
        """
        vizinhos = []

        for i in range(len(params)):
            tipo = tipos[i]

            if tipo == 'int':
                # Para inteiros, incrementa e decrementa
                step = max(1, int(step_size))

                # Vizinho +
                params_mais = params.copy()
                novo_valor = params[i] + step
                if novo_valor <= limite_max:
                    params_mais[i] = novo_valor
                    vizinhos.append(params_mais)

                # Vizinho -
                params_menos = params.copy()
                novo_valor = params[i] - step
                if novo_valor >= limite_min:
                    params_menos[i] = novo_valor
                    vizinhos.append(params_menos)

            elif tipo == 'float':
                # Para floats, incrementa e decrementa
                step = float(step_size)

                # Vizinho +
                params_mais = params.copy()
                novo_valor = round(params[i] + step, 6)
                if novo_valor <= limite_max:
                    params_mais[i] = novo_valor
                    vizinhos.append(params_mais)

                # Vizinho -
                params_menos = params.copy()
                novo_valor = round(params[i] - step, 6)
                if novo_valor >= limite_min:
                    params_menos[i] = novo_valor
                    vizinhos.append(params_menos)

            elif tipo == 'str':
                # Para strings, testa variacoes conhecidas
                # Valores comuns: baixo/medio/alto, true/false, etc
                opcoes_comuns = {
                    'baixo': ['medio', 'alto'],
                    'medio': ['baixo', 'alto'],
                    'alto': ['baixo', 'medio'],
                    'true': ['false'],
                    'false': ['true'],
                    'sim': ['nao'],
                    'nao': ['sim'],
                    'yes': ['no'],
                    'no': ['yes']
                }

                valor_atual = str(params[i]).lower()
                if valor_atual in opcoes_comuns:
                    # Gera vizinhos com as opcoes alternativas
                    for opcao in opcoes_comuns[valor_atual]:
                        params_alt = params.copy()
                        params_alt[i] = opcao
                        vizinhos.append(params_alt)

        return vizinhos

    def otimizar(self,
                 params_iniciais: List,
                 tipos: List[str],
                 step_inicial: Union[int, float] = 20,
                 step_minimo: Union[int, float] = 1,
                 fator_reducao: float = 0.5,
                 max_iter: int = 1000,
                 limite_min: Union[int, float] = 1,
                 limite_max: Union[int, float] = 100):
        """
        Executa o Pattern Search

        Args:
            params_iniciais: Parametros iniciais
            tipos: Tipos de cada parametro
            step_inicial: Tamanho inicial do passo
            step_minimo: Tamanho minimo do passo
            fator_reducao: Fator de reducao do passo quando nao ha melhora
            max_iter: Numero maximo de iteracoes
            limite_min: Limite inferior para valores numericos
            limite_max: Limite superior para valores numericos
        """
        print("\n" + "="*70)
        print("PATTERN SEARCH OPTIMIZER")
        print("="*70)
        print(f"Modo: {self.modo.upper()}")
        print(f"Parametros iniciais: {params_iniciais}")
        print(f"Tipos detectados: {tipos}")
        print(f"Numero de variaveis: {len(params_iniciais)}")
        print(f"Step inicial: {step_inicial}")
        print(f"Limites: [{limite_min}, {limite_max}]")
        print("="*70 + "\n")

        # Inicializa
        params_atual = params_iniciais.copy()
        step_size = step_inicial
        iteracao = 0
        iteracoes_sem_melhoria = 0
        max_iter_sem_melhoria = 10  # Para no maximo 10 iteracoes sem melhoria

        # Avalia ponto inicial
        print(f"Avaliando ponto inicial...")
        valor_atual = self.executar_modelo(params_atual)
        print(f"Valor inicial: {valor_atual:.6f}\n")

        # Loop principal
        while iteracao < max_iter and step_size >= step_minimo:
            iteracao += 1
            print(f"\n--- Iteracao {iteracao} | Step: {step_size} ---")

            # Gera vizinhos
            vizinhos = self.gerar_vizinhos(params_atual, tipos, step_size, limite_min, limite_max)

            if len(vizinhos) == 0:
                print("> Nenhum vizinho valido.")

                # Se ja estamos no step minimo e nao ha vizinhos, pare
                if step_size <= step_minimo:
                    print("> Step no minimo e sem vizinhos validos. Parando.")
                    break

                print("> Reduzindo step.")
                step_size = step_size * fator_reducao
                if 'int' in tipos:
                    step_size = max(step_minimo, int(step_size))
                iteracoes_sem_melhoria += 1

                # Se passou muitas iteracoes sem melhoria, pare
                if iteracoes_sem_melhoria >= max_iter_sem_melhoria:
                    print(f"> Sem melhoria por {max_iter_sem_melhoria} iteracoes. Parando.")
                    break

                continue

            # Avalia vizinhos
            melhor_vizinho = None
            melhor_valor_vizinho = valor_atual

            for viz in vizinhos:
                valor_viz = self.executar_modelo(viz)
                # Compara baseado no modo
                if self.modo == 'maximizar':
                    if valor_viz > melhor_valor_vizinho:
                        melhor_valor_vizinho = valor_viz
                        melhor_vizinho = viz
                else:  # minimizar
                    if valor_viz < melhor_valor_vizinho:
                        melhor_valor_vizinho = valor_viz
                        melhor_vizinho = viz

            # Se encontrou um vizinho melhor, move para ele
            if melhor_vizinho is not None:
                params_atual = melhor_vizinho
                valor_atual = melhor_valor_vizinho
                print(f"> Movendo para vizinho melhor: {params_atual} = {valor_atual:.6f}")
                iteracoes_sem_melhoria = 0  # Reseta contador
            else:
                # Nao encontrou melhoria, reduz o step
                iteracoes_sem_melhoria += 1

                # Se ja estamos no step minimo e nao ha melhoria, pare
                if step_size <= step_minimo:
                    print("> Step no minimo e sem melhoria. Parando.")
                    break

                step_size = step_size * fator_reducao
                if 'int' in tipos:
                    step_size = max(step_minimo, int(step_size))
                print(f"> Nenhuma melhoria. Reduzindo step para: {step_size}")

                # Se passou muitas iteracoes sem melhoria, pare
                if iteracoes_sem_melhoria >= max_iter_sem_melhoria:
                    print(f"> Sem melhoria por {max_iter_sem_melhoria} iteracoes consecutivas. Parando.")
                    break

        # Resultado final
        print("\n" + "="*70)
        print("OTIMIZACAO CONCLUIDA")
        print("="*70)
        print(f"Total de execucoes: {self.total_execucoes}")
        print(f"Iteracoes realizadas: {iteracao}")
        tipo_otimo = "maximo" if self.modo == 'maximizar' else "minimo"
        print(f"Valor {tipo_otimo} encontrado: {self.melhor_valor:.6f}")
        print(f"Parametros otimos: {self.melhor_params}")
        print("="*70 + "\n")

        return self.melhor_params, self.melhor_valor

    def salvar_relatorio(self, arquivo: str = "resultado_pattern_search.txt"):
        """Salva relatorio da otimizacao"""
        with open(arquivo, 'w', encoding='utf-8') as f:
            f.write("RELATORIO DE OTIMIZACAO - PATTERN SEARCH\n")
            f.write("="*70 + "\n\n")
            f.write(f"Valor Maximo: {self.melhor_valor:.6f}\n")
            f.write(f"Parametros Otimos: {self.melhor_params}\n")
            f.write(f"Total de Execucoes: {self.total_execucoes}\n\n")
            f.write("="*70 + "\n")
            f.write("HISTORICO COMPLETO\n")
            f.write("="*70 + "\n\n")
            for i, (params, valor) in enumerate(self.historico, 1):
                f.write(f"{i}. {params} -> {valor:.6f}\n")

        print(f"[OK] Relatorio salvo em: {arquivo}")


class GeneticAlgorithm:
    """Implementacao do Algoritmo Genetico"""

    def __init__(self, comando_modelo: str, modo: str = 'maximizar'):
        """
        Inicializa o Algoritmo Genetico

        Args:
            comando_modelo: Comando do executavel (ex: 'modelo10.exe')
            modo: 'maximizar' ou 'minimizar' (padrao: 'maximizar')
        """
        self.comando_modelo = comando_modelo
        self.modo = modo.lower()
        self.melhor_valor = float('-inf') if self.modo == 'maximizar' else float('inf')
        self.melhor_params = None
        self.historico = []
        self.total_execucoes = 0

    def detectar_padrao(self, padrao: str) -> Tuple[List, List[str]]:
        """Detecta automaticamente o padrao e tipos das variaveis (mesmo do PatternSearch)"""
        valores_str = padrao.strip().split()
        valores = []
        tipos = []

        for val in valores_str:
            # 1. Tenta INT primeiro
            try:
                valor_int = int(val)
                if valor_int == 0:
                    valor_int = 1
                valores.append(valor_int)
                tipos.append('int')
                continue
            except ValueError:
                pass

            # 2. Tenta FLOAT
            try:
                valor_float = float(val)
                if valor_float == 0.0:
                    valor_float = 1.0
                valores.append(valor_float)
                tipos.append('float')
                continue
            except ValueError:
                pass

            # 3. E TEXTUAL
            valores.append(val.lower())
            tipos.append('str')

        return valores, tipos

    def executar_modelo(self, params: List) -> float:
        """Executa o modelo com os parametros dados"""
        try:
            params_str = [str(p) for p in params]
            resultado = subprocess.run(
                [self.comando_modelo] + params_str,
                capture_output=True,
                text=True,
                timeout=30
            )
            saida = resultado.stdout.strip()

            if ":" in saida:
                valor = float(saida.split(":")[-1].strip())
            else:
                valor = float(saida)

            self.total_execucoes += 1

            # Atualiza melhor resultado
            melhorou = False
            if self.modo == 'maximizar':
                if valor > self.melhor_valor:
                    melhorou = True
            else:
                if valor < self.melhor_valor:
                    melhorou = True

            if melhorou:
                self.melhor_valor = valor
                self.melhor_params = params.copy()
                tipo_otimo = "maximo" if self.modo == 'maximizar' else "minimo"
                print(f"[OK] Novo {tipo_otimo}: {valor:.6f} com {params}")

            self.historico.append((params.copy(), valor))
            return valor

        except ValueError as e:
            # Erro ao converter para float - provavelmente o modelo retornou uma mensagem de erro
            print(f"[ERRO] Modelo retornou mensagem de erro com parametros {params}:")
            print(f"       '{resultado.stdout.strip()}'")
            return float('-inf') if self.modo == 'maximizar' else float('inf')
        except Exception as e:
            print(f"[ERRO] Erro ao executar modelo com parametros {params}: {e}")
            # Retorna o pior valor possivel
            return float('-inf') if self.modo == 'maximizar' else float('inf')

    def gerar_individuo(self, tipos: List[str], limite_min: float, limite_max: float,
                        valores_referencia: List = None) -> List:
        """Gera um individuo aleatorio"""
        import random
        individuo = []

        for i, tipo in enumerate(tipos):
            if tipo == 'int':
                individuo.append(random.randint(int(limite_min), int(limite_max)))
            elif tipo == 'float':
                individuo.append(round(random.uniform(limite_min, limite_max), 6))
            elif tipo == 'str':
                # Para strings, usa o valor de referencia (do padrao inicial)
                # Isso evita gerar strings que o modelo nao reconhece
                if valores_referencia and i < len(valores_referencia):
                    individuo.append(valores_referencia[i])
                else:
                    individuo.append('medio')  # Valor padrao seguro

        return individuo

    def crossover(self, pai1: List, pai2: List, tipos: List[str]) -> Tuple[List, List]:
        """Realiza crossover entre dois pais"""
        import random
        ponto_corte = random.randint(1, len(pai1) - 1)

        filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
        filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]

        return filho1, filho2

    def mutacao(self, individuo: List, tipos: List[str], taxa_mutacao: float,
                limite_min: float, limite_max: float) -> List:
        """Realiza mutacao em um individuo"""
        import random
        individuo_mutado = individuo.copy()

        for i in range(len(individuo_mutado)):
            if random.random() < taxa_mutacao:
                tipo = tipos[i]

                if tipo == 'int':
                    # Mutacao: adiciona um valor aleatorio pequeno
                    delta = random.randint(-10, 10)
                    novo_valor = individuo_mutado[i] + delta
                    individuo_mutado[i] = max(int(limite_min), min(int(limite_max), novo_valor))

                elif tipo == 'float':
                    # Mutacao: adiciona um valor aleatorio pequeno
                    delta = random.uniform(-5.0, 5.0)
                    novo_valor = individuo_mutado[i] + delta
                    individuo_mutado[i] = round(max(limite_min, min(limite_max, novo_valor)), 6)

                elif tipo == 'str':
                    # Para strings, NAO faz mutacao
                    # Cada modelo aceita valores diferentes, entao mantemos o valor original
                    # A exploracao de strings e feita apenas pelo crossover
                    pass

        return individuo_mutado

    def selecao_torneio(self, populacao: List[List], fitness: List[float], tamanho_torneio: int = 3) -> List:
        """Seleciona um individuo usando torneio"""
        import random
        torneio_indices = random.sample(range(len(populacao)), min(tamanho_torneio, len(populacao)))

        if self.modo == 'maximizar':
            melhor_idx = max(torneio_indices, key=lambda i: fitness[i])
        else:
            melhor_idx = min(torneio_indices, key=lambda i: fitness[i])

        return populacao[melhor_idx]

    def otimizar(self, params_iniciais: List, tipos: List[str],
                 tamanho_populacao: int = 50,
                 num_geracoes: int = 100,
                 taxa_crossover: float = 0.8,
                 taxa_mutacao: float = 0.1,
                 limite_min: float = 1,
                 limite_max: float = 100):
        """
        Executa o Algoritmo Genetico

        Args:
            params_iniciais: Parametros iniciais (usado como referencia)
            tipos: Tipos de cada parametro
            tamanho_populacao: Tamanho da populacao
            num_geracoes: Numero de geracoes
            taxa_crossover: Taxa de crossover (0-1)
            taxa_mutacao: Taxa de mutacao (0-1)
            limite_min: Limite inferior para valores numericos
            limite_max: Limite superior para valores numericos
        """
        import random

        print("\n" + "="*70)
        print("GENETIC ALGORITHM OPTIMIZER")
        print("="*70)
        print(f"Modo: {self.modo.upper()}")
        print(f"Tipos detectados: {tipos}")
        print(f"Numero de variaveis: {len(params_iniciais)}")
        print(f"Parametros iniciais: {params_iniciais}")
        print(f"Tamanho da populacao: {tamanho_populacao}")
        print(f"Numero de geracoes: {num_geracoes}")
        print(f"Taxa de crossover: {taxa_crossover}")
        print(f"Taxa de mutacao: {taxa_mutacao}")
        print(f"Limites: [{limite_min}, {limite_max}]")

        # Aviso sobre strings
        if 'str' in tipos:
            print("\n[AVISO] Parametros tipo STRING nao sao explorados pelo GA.")
            print("        Todos os individuos usarao os valores string do padrao inicial.")
            print("        Se houver erros, verifique se os valores string estao corretos.")
            print("        Para explorar strings, use Pattern Search (opcao 1).")

        print("="*70 + "\n")

        # Inicializa populacao
        print("Gerando populacao inicial...")
        populacao = [self.gerar_individuo(tipos, limite_min, limite_max, params_iniciais)
                     for _ in range(tamanho_populacao)]

        # Adiciona o individuo inicial na populacao
        populacao[0] = params_iniciais.copy()

        # Loop de evolucao
        for geracao in range(num_geracoes):
            print(f"\n--- Geracao {geracao + 1}/{num_geracoes} ---")

            # Avalia fitness de toda a populacao
            fitness = [self.executar_modelo(ind) for ind in populacao]

            # Mostra melhor da geracao
            if self.modo == 'maximizar':
                melhor_idx = max(range(len(fitness)), key=lambda i: fitness[i])
            else:
                melhor_idx = min(range(len(fitness)), key=lambda i: fitness[i])

            print(f"> Melhor da geracao: {fitness[melhor_idx]:.6f}")

            # Nova populacao
            nova_populacao = []

            # Elitismo: mantém o melhor individuo
            nova_populacao.append(populacao[melhor_idx].copy())

            # Gera o restante da populacao
            while len(nova_populacao) < tamanho_populacao:
                # Selecao
                pai1 = self.selecao_torneio(populacao, fitness)
                pai2 = self.selecao_torneio(populacao, fitness)

                # Crossover
                if random.random() < taxa_crossover:
                    filho1, filho2 = self.crossover(pai1, pai2, tipos)
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()

                # Mutacao
                filho1 = self.mutacao(filho1, tipos, taxa_mutacao, limite_min, limite_max)
                filho2 = self.mutacao(filho2, tipos, taxa_mutacao, limite_min, limite_max)

                nova_populacao.append(filho1)
                if len(nova_populacao) < tamanho_populacao:
                    nova_populacao.append(filho2)

            populacao = nova_populacao

        # Resultado final
        print("\n" + "="*70)
        print("OTIMIZACAO CONCLUIDA")
        print("="*70)
        print(f"Total de execucoes: {self.total_execucoes}")
        print(f"Geracoes realizadas: {num_geracoes}")
        tipo_otimo = "maximo" if self.modo == 'maximizar' else "minimo"
        print(f"Valor {tipo_otimo} encontrado: {self.melhor_valor:.6f}")
        print(f"Parametros otimos: {self.melhor_params}")
        print("="*70 + "\n")

        return self.melhor_params, self.melhor_valor

    def salvar_relatorio(self, arquivo: str = "resultado_genetic_algorithm.txt"):
        """Salva relatorio da otimizacao"""
        with open(arquivo, 'w', encoding='utf-8') as f:
            f.write("RELATORIO DE OTIMIZACAO - GENETIC ALGORITHM\n")
            f.write("="*70 + "\n\n")
            tipo_otimo = "Maximo" if self.modo == 'maximizar' else "Minimo"
            f.write(f"Valor {tipo_otimo}: {self.melhor_valor:.6f}\n")
            f.write(f"Parametros Otimos: {self.melhor_params}\n")
            f.write(f"Total de Execucoes: {self.total_execucoes}\n\n")
            f.write("="*70 + "\n")
            f.write("HISTORICO COMPLETO\n")
            f.write("="*70 + "\n\n")
            for i, (params, valor) in enumerate(self.historico, 1):
                f.write(f"{i}. {params} -> {valor:.6f}\n")

        print(f"[OK] Relatorio salvo em: {arquivo}")


class ParticleSwarm:
    """Implementacao do Particle Swarm Optimization (PSO)"""

    def __init__(self, comando_modelo: str, modo: str = 'maximizar'):
        """
        Inicializa o PSO

        Args:
            comando_modelo: Comando do executavel (ex: 'modelo10.exe')
            modo: 'maximizar' ou 'minimizar' (padrao: 'maximizar')
        """
        self.comando_modelo = comando_modelo
        self.modo = modo.lower()
        self.melhor_valor = float('-inf') if self.modo == 'maximizar' else float('inf')
        self.melhor_params = None
        self.historico = []
        self.total_execucoes = 0

    def detectar_padrao(self, padrao: str) -> Tuple[List, List[str]]:
        """Detecta automaticamente o padrao e tipos das variaveis"""
        valores_str = padrao.strip().split()
        valores = []
        tipos = []

        for val in valores_str:
            try:
                valor_int = int(val)
                if valor_int == 0:
                    valor_int = 1
                valores.append(valor_int)
                tipos.append('int')
                continue
            except ValueError:
                pass

            try:
                valor_float = float(val)
                if valor_float == 0.0:
                    valor_float = 1.0
                valores.append(valor_float)
                tipos.append('float')
                continue
            except ValueError:
                pass

            valores.append(val.lower())
            tipos.append('str')

        return valores, tipos

    def executar_modelo(self, params: List) -> float:
        """Executa o modelo com os parametros dados"""
        try:
            params_str = [str(p) for p in params]
            resultado = subprocess.run(
                [self.comando_modelo] + params_str,
                capture_output=True,
                text=True,
                timeout=30
            )
            saida = resultado.stdout.strip()

            if ":" in saida:
                valor = float(saida.split(":")[-1].strip())
            else:
                valor = float(saida)

            self.total_execucoes += 1

            # Atualiza melhor resultado global
            melhorou = False
            if self.modo == 'maximizar':
                if valor > self.melhor_valor:
                    melhorou = True
            else:
                if valor < self.melhor_valor:
                    melhorou = True

            if melhorou:
                self.melhor_valor = valor
                self.melhor_params = params.copy()
                tipo_otimo = "maximo" if self.modo == 'maximizar' else "minimo"
                print(f"[OK] Novo {tipo_otimo}: {valor:.6f} com {params}")

            self.historico.append((params.copy(), valor))
            return valor

        except ValueError as e:
            # Erro ao converter para float - provavelmente o modelo retornou uma mensagem de erro
            print(f"[ERRO] Modelo retornou mensagem de erro com parametros {params}:")
            print(f"       '{resultado.stdout.strip()}'")
            return float('-inf') if self.modo == 'maximizar' else float('inf')
        except Exception as e:
            print(f"[ERRO] Erro ao executar modelo com parametros {params}: {e}")
            return float('-inf') if self.modo == 'maximizar' else float('inf')

    def gerar_particula(self, tipos: List[str], limite_min: float, limite_max: float,
                        valores_referencia: List = None) -> List:
        """Gera uma particula (posicao) aleatoria"""
        import random
        particula = []

        for i, tipo in enumerate(tipos):
            if tipo == 'int':
                particula.append(random.randint(int(limite_min), int(limite_max)))
            elif tipo == 'float':
                particula.append(round(random.uniform(limite_min, limite_max), 6))
            elif tipo == 'str':
                # Para strings, usa o valor de referencia (do padrao inicial)
                # Isso evita gerar strings que o modelo nao reconhece
                if valores_referencia and i < len(valores_referencia):
                    particula.append(valores_referencia[i])
                else:
                    particula.append('medio')  # Valor padrao seguro

        return particula

    def gerar_velocidade(self, tipos: List[str]) -> List:
        """Gera uma velocidade inicial (pequena)"""
        import random
        velocidade = []

        for tipo in tipos:
            if tipo == 'int':
                velocidade.append(random.randint(-5, 5))
            elif tipo == 'float':
                velocidade.append(round(random.uniform(-1.0, 1.0), 6))
            elif tipo == 'str':
                velocidade.append(0)  # Para strings, velocidade nao se aplica diretamente

        return velocidade

    def atualizar_velocidade(self, velocidade: List, posicao: List, pBest: List, gBest: List,
                            tipos: List[str], w: float, c1: float, c2: float) -> List:
        """Atualiza a velocidade de uma particula"""
        import random
        nova_velocidade = []

        for i in range(len(velocidade)):
            tipo = tipos[i]

            if tipo in ['int', 'float']:
                # Formula PSO: v = w*v + c1*r1*(pBest - pos) + c2*r2*(gBest - pos)
                r1 = random.random()
                r2 = random.random()

                componente_inercia = w * velocidade[i]
                componente_cognitiva = c1 * r1 * (pBest[i] - posicao[i])
                componente_social = c2 * r2 * (gBest[i] - posicao[i])

                v_nova = componente_inercia + componente_cognitiva + componente_social

                # Limita a velocidade maxima
                v_max = 10.0 if tipo == 'float' else 10
                v_nova = max(-v_max, min(v_max, v_nova))

                if tipo == 'int':
                    v_nova = int(v_nova)
                else:
                    v_nova = round(v_nova, 6)

                nova_velocidade.append(v_nova)

            elif tipo == 'str':
                # Para strings, NAO atualiza velocidade
                # Mantém sempre 0 para não mudar o valor
                nova_velocidade.append(0)

        return nova_velocidade

    def atualizar_posicao(self, posicao: List, velocidade: List, tipos: List[str],
                         pBest: List, gBest: List, limite_min: float, limite_max: float) -> List:
        """Atualiza a posicao de uma particula"""
        import random
        nova_posicao = []

        for i in range(len(posicao)):
            tipo = tipos[i]

            if tipo == 'int':
                novo_valor = posicao[i] + velocidade[i]
                novo_valor = max(int(limite_min), min(int(limite_max), int(novo_valor)))
                nova_posicao.append(novo_valor)

            elif tipo == 'float':
                novo_valor = posicao[i] + velocidade[i]
                novo_valor = round(max(limite_min, min(limite_max, novo_valor)), 6)
                nova_posicao.append(novo_valor)

            elif tipo == 'str':
                # Para strings, NUNCA muda - mantém o valor atual
                # Cada modelo aceita valores diferentes, entao nao podemos mudar aleatoriamente
                nova_posicao.append(posicao[i])

        return nova_posicao

    def otimizar(self, params_iniciais: List, tipos: List[str],
                 num_particulas: int = 30,
                 num_iteracoes: int = 100,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 limite_min: float = 1,
                 limite_max: float = 100):
        """
        Executa o PSO

        Args:
            params_iniciais: Parametros iniciais
            tipos: Tipos de cada parametro
            num_particulas: Numero de particulas no enxame
            num_iteracoes: Numero de iteracoes
            w: Peso de inercia
            c1: Constante cognitiva
            c2: Constante social
            limite_min: Limite inferior
            limite_max: Limite superior
        """
        import random

        print("\n" + "="*70)
        print("PARTICLE SWARM OPTIMIZATION")
        print("="*70)
        print(f"Modo: {self.modo.upper()}")
        print(f"Tipos detectados: {tipos}")
        print(f"Numero de variaveis: {len(params_iniciais)}")
        print(f"Parametros iniciais: {params_iniciais}")
        print(f"Numero de particulas: {num_particulas}")
        print(f"Numero de iteracoes: {num_iteracoes}")
        print(f"Peso de inercia (w): {w}")
        print(f"Constante cognitiva (c1): {c1}")
        print(f"Constante social (c2): {c2}")
        print(f"Limites: [{limite_min}, {limite_max}]")

        # Aviso sobre strings
        if 'str' in tipos:
            print("\n[AVISO] Parametros tipo STRING nao sao explorados pelo PSO.")
            print("        Todas as particulas usarao os valores string do padrao inicial.")
            print("        Se houver erros, verifique se os valores string estao corretos.")
            print("        Para explorar strings, use Pattern Search (opcao 1).")

        print("="*70 + "\n")

        # Inicializa enxame
        print("Gerando enxame inicial...")
        particulas = [self.gerar_particula(tipos, limite_min, limite_max, params_iniciais)
                      for _ in range(num_particulas)]
        velocidades = [self.gerar_velocidade(tipos) for _ in range(num_particulas)]

        # Primeira particula eh a inicial
        particulas[0] = params_iniciais.copy()

        # Avalia fitness inicial
        fitness = [self.executar_modelo(p) for p in particulas]

        # Inicializa pBest (melhor pessoal de cada particula)
        pBest = [particulas[i].copy() for i in range(num_particulas)]
        pBest_fitness = fitness.copy()

        # Inicializa gBest (melhor global)
        if self.modo == 'maximizar':
            melhor_idx = max(range(len(fitness)), key=lambda i: fitness[i])
        else:
            melhor_idx = min(range(len(fitness)), key=lambda i: fitness[i])

        gBest = particulas[melhor_idx].copy()
        gBest_fitness = fitness[melhor_idx]

        # Loop principal
        for iteracao in range(num_iteracoes):
            print(f"\n--- Iteracao {iteracao + 1}/{num_iteracoes} ---")

            for i in range(num_particulas):
                # Atualiza velocidade
                velocidades[i] = self.atualizar_velocidade(
                    velocidades[i], particulas[i], pBest[i], gBest,
                    tipos, w, c1, c2
                )

                # Atualiza posicao
                particulas[i] = self.atualizar_posicao(
                    particulas[i], velocidades[i], tipos, pBest[i], gBest,
                    limite_min, limite_max
                )

                # Avalia nova posicao
                fitness[i] = self.executar_modelo(particulas[i])

                # Atualiza pBest
                melhorou_pessoal = False
                if self.modo == 'maximizar':
                    if fitness[i] > pBest_fitness[i]:
                        melhorou_pessoal = True
                else:
                    if fitness[i] < pBest_fitness[i]:
                        melhorou_pessoal = True

                if melhorou_pessoal:
                    pBest[i] = particulas[i].copy()
                    pBest_fitness[i] = fitness[i]

                    # Atualiza gBest
                    melhorou_global = False
                    if self.modo == 'maximizar':
                        if fitness[i] > gBest_fitness:
                            melhorou_global = True
                    else:
                        if fitness[i] < gBest_fitness:
                            melhorou_global = True

                    if melhorou_global:
                        gBest = particulas[i].copy()
                        gBest_fitness = fitness[i]

            print(f"> Melhor global: {gBest_fitness:.6f}")

        # Resultado final
        print("\n" + "="*70)
        print("OTIMIZACAO CONCLUIDA")
        print("="*70)
        print(f"Total de execucoes: {self.total_execucoes}")
        print(f"Iteracoes realizadas: {num_iteracoes}")
        tipo_otimo = "maximo" if self.modo == 'maximizar' else "minimo"
        print(f"Valor {tipo_otimo} encontrado: {self.melhor_valor:.6f}")
        print(f"Parametros otimos: {self.melhor_params}")
        print("="*70 + "\n")

        return self.melhor_params, self.melhor_valor

    def salvar_relatorio(self, arquivo: str = "resultado_particle_swarm.txt"):
        """Salva relatorio da otimizacao"""
        with open(arquivo, 'w', encoding='utf-8') as f:
            f.write("RELATORIO DE OTIMIZACAO - PARTICLE SWARM OPTIMIZATION\n")
            f.write("="*70 + "\n\n")
            tipo_otimo = "Maximo" if self.modo == 'maximizar' else "Minimo"
            f.write(f"Valor {tipo_otimo}: {self.melhor_valor:.6f}\n")
            f.write(f"Parametros Otimos: {self.melhor_params}\n")
            f.write(f"Total de Execucoes: {self.total_execucoes}\n\n")
            f.write("="*70 + "\n")
            f.write("HISTORICO COMPLETO\n")
            f.write("="*70 + "\n\n")
            for i, (params, valor) in enumerate(self.historico, 1):
                f.write(f"{i}. {params} -> {valor:.6f}\n")

        print(f"[OK] Relatorio salvo em: {arquivo}")


def mostrar_menu():
    """Mostra o menu principal"""
    print("\n" + "="*70)
    print("        SISTEMA DE OTIMIZACAO AUTOMATICA")
    print("="*70)
    print("\nMETODOS DE OTIMIZACAO DISPONIVEIS:\n")
    print("  [1] Pattern Search (Busca por Padroes)")
    print("  [2] Genetic Algorithm (Algoritmo Genetico)")
    print("  [3] Particle Swarm (Otimizacao por Enxame de Particulas)")
    print("  [4] Simulated Annealing (Em breve...)")
    print("\n  [0] Sair")
    print("\n" + "="*70)


def executar_pattern_search():
    """Executa o Pattern Search com configuracao automatica"""

    print("\n" + "="*70)
    print("CONFIGURACAO DO PATTERN SEARCH")
    print("="*70 + "\n")

    # 1. Entrada completa: comando + padrao
    print("1. Digite o COMANDO + PADRAO (tudo em uma linha):")
    print("   Exemplos:")
    print("   - modelo10.exe baixo 1 1 1 1 1 1 1 1 1")
    print("   - ./modelo10.exe baixo 1 1.5 true 1")
    print("   - meu_modelo.exe 1 1 1 1 1 1 1 1")
    print("\n   O sistema detecta automaticamente:")
    print("   - Comando do executavel (primeira palavra)")
    print("   - Quantidade de variaveis (demais palavras)")
    print("   - Tipos: INT, FLOAT ou STRING")
    print("\n   NOTA: Valores 0 ou 0.0 serao convertidos para 1 ou 1.0")

    entrada_completa = input("\n   Digite: ").strip()

    # Se nao digitou nada, usa padrao
    if not entrada_completa:
        entrada_completa = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
        print(f"   [PADRAO] Usando: {entrada_completa}")

    # Separar comando do padrao
    partes = entrada_completa.split()
    if len(partes) < 2:
        print("\n   [ERRO] Formato invalido! Use: comando param1 param2 ...")
        print("   Exemplo: modelo10.exe baixo 1 1 1 1 1 1 1 1 1")
        return False

    comando_modelo = partes[0]
    padrao = " ".join(partes[1:])

    print(f"\n   [OK] Comando detectado: {comando_modelo}")
    print(f"   [OK] Padrao detectado: {padrao}")

    # Criar otimizador temporario para detectar padrao
    optimizer_temp = PatternSearch(comando_modelo)

    # Detectar padrao
    params_iniciais, tipos = optimizer_temp.detectar_padrao(padrao)

    print(f"\n   [OK] Analise do padrao:")
    print(f"   - Quantidade de variaveis: {len(params_iniciais)}")
    print(f"   - Tipos: {tipos}")
    print(f"   - Valores iniciais: {params_iniciais}")

    # 2. Escolher modo: maximizar ou minimizar
    print("\n2. Modo de otimizacao:")
    print("   [1] Maximizar (buscar maior valor)")
    print("   [2] Minimizar (buscar menor valor)")
    modo_input = input("   Escolha [1]: ").strip()
    modo = 'minimizar' if modo_input == '2' else 'maximizar'
    print(f"   [OK] Modo selecionado: {modo.upper()}")

    # Criar otimizador com o modo escolhido
    optimizer = PatternSearch(comando_modelo, modo=modo)

    # 3. Configuracoes de otimizacao
    print("\n3. Configuracoes de otimizacao (pressione ENTER para usar padrao):")

    try:
        step_inicial_input = input("   Step inicial [20]: ").strip()
        step_inicial = float(step_inicial_input) if step_inicial_input else 20

        step_minimo_input = input("   Step minimo [1]: ").strip()
        step_minimo = float(step_minimo_input) if step_minimo_input else 1

        max_iter_input = input("   Maximo de iteracoes [1000]: ").strip()
        max_iter = int(max_iter_input) if max_iter_input else 1000

        limite_min_input = input("   Limite minimo [1]: ").strip()
        limite_min = float(limite_min_input) if limite_min_input else 1

        limite_max_input = input("   Limite maximo [100]: ").strip()
        limite_max = float(limite_max_input) if limite_max_input else 100

    except ValueError:
        print("   [AVISO] Valor invalido. Usando configuracoes padrao.")
        step_inicial = 20
        step_minimo = 1
        max_iter = 1000
        limite_min = 1
        limite_max = 100

    # 3. Executa otimizacao
    print("\n" + "="*70)
    print("INICIANDO OTIMIZACAO AUTOMATICA...")
    print("="*70)
    objetivo = "MAIOR VALOR" if modo == 'maximizar' else "MENOR VALOR"
    print(f"\nO algoritmo tentara encontrar o {objetivo} automaticamente.")
    print("Pressione Ctrl+C para interromper a qualquer momento.\n")

    try:
        melhor_params, melhor_valor = optimizer.otimizar(
            params_iniciais=params_iniciais,
            tipos=tipos,
            step_inicial=step_inicial,
            step_minimo=step_minimo,
            max_iter=max_iter,
            limite_min=limite_min,
            limite_max=limite_max
        )

        # Salva relatorio
        optimizer.salvar_relatorio()

        # Mostra comando para reproduzir
        print("\n" + "-"*70)
        print("COMANDO PARA REPRODUZIR O MELHOR RESULTADO:")
        params_str = " ".join(str(p) for p in melhor_params)
        print(f"\n{comando_modelo} {params_str}")
        print("\n" + "-"*70 + "\n")

        return True

    except KeyboardInterrupt:
        print("\n\n[AVISO] Otimizacao interrompida pelo usuario.")
        if optimizer.melhor_params:
            print(f"\nMelhor resultado ate agora:")
            print(f"Valor: {optimizer.melhor_valor:.6f}")
            print(f"Parametros: {optimizer.melhor_params}")
        return False

    except Exception as e:
        print(f"\n[ERRO] Erro durante otimizacao: {e}")
        import traceback
        traceback.print_exc()
        return False


def executar_genetic_algorithm():
    """Executa o Genetic Algorithm com configuracao automatica"""

    print("\n" + "="*70)
    print("CONFIGURACAO DO GENETIC ALGORITHM")
    print("="*70 + "\n")

    # 1. Entrada completa: comando + padrao
    print("1. Digite o COMANDO + PADRAO (tudo em uma linha):")
    print("   Exemplos:")
    print("   - modelo10.exe baixo 1 1 1 1 1 1 1 1 1")
    print("   - ./modelo10.exe baixo 1 1.5 true 1")
    print("   - meu_modelo.exe 1 1 1 1 1 1 1 1")

    entrada_completa = input("\n   Digite: ").strip()

    if not entrada_completa:
        entrada_completa = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
        print(f"   [PADRAO] Usando: {entrada_completa}")

    partes = entrada_completa.split()
    if len(partes) < 2:
        print("\n   [ERRO] Formato invalido! Use: comando param1 param2 ...")
        return False

    comando_modelo = partes[0]
    padrao = " ".join(partes[1:])

    print(f"\n   [OK] Comando detectado: {comando_modelo}")
    print(f"   [OK] Padrao detectado: {padrao}")

    # Criar otimizador temporario para detectar padrao
    optimizer_temp = GeneticAlgorithm(comando_modelo)
    params_iniciais, tipos = optimizer_temp.detectar_padrao(padrao)

    print(f"\n   [OK] Analise do padrao:")
    print(f"   - Quantidade de variaveis: {len(params_iniciais)}")
    print(f"   - Tipos: {tipos}")
    print(f"   - Valores iniciais: {params_iniciais}")

    # 2. Escolher modo: maximizar ou minimizar
    print("\n2. Modo de otimizacao:")
    print("   [1] Maximizar (buscar maior valor)")
    print("   [2] Minimizar (buscar menor valor)")
    modo_input = input("   Escolha [1]: ").strip()
    modo = 'minimizar' if modo_input == '2' else 'maximizar'
    print(f"   [OK] Modo selecionado: {modo.upper()}")

    # Criar otimizador com o modo escolhido
    optimizer = GeneticAlgorithm(comando_modelo, modo=modo)

    # 3. Configuracoes de otimizacao
    print("\n3. Configuracoes de otimizacao (pressione ENTER para usar padrao):")

    try:
        tamanho_pop_input = input("   Tamanho da populacao [50]: ").strip()
        tamanho_populacao = int(tamanho_pop_input) if tamanho_pop_input else 50

        num_ger_input = input("   Numero de geracoes [100]: ").strip()
        num_geracoes = int(num_ger_input) if num_ger_input else 100

        taxa_cross_input = input("   Taxa de crossover [0.8]: ").strip()
        taxa_crossover = float(taxa_cross_input) if taxa_cross_input else 0.8

        taxa_mut_input = input("   Taxa de mutacao [0.1]: ").strip()
        taxa_mutacao = float(taxa_mut_input) if taxa_mut_input else 0.1

        limite_min_input = input("   Limite minimo [1]: ").strip()
        limite_min = float(limite_min_input) if limite_min_input else 1

        limite_max_input = input("   Limite maximo [100]: ").strip()
        limite_max = float(limite_max_input) if limite_max_input else 100

    except ValueError:
        print("   [AVISO] Valor invalido. Usando configuracoes padrao.")
        tamanho_populacao = 50
        num_geracoes = 100
        taxa_crossover = 0.8
        taxa_mutacao = 0.1
        limite_min = 1
        limite_max = 100

    # 4. Executa otimizacao
    print("\n" + "="*70)
    print("INICIANDO OTIMIZACAO AUTOMATICA...")
    print("="*70)
    objetivo = "MAIOR VALOR" if modo == 'maximizar' else "MENOR VALOR"
    print(f"\nO algoritmo tentara encontrar o {objetivo} automaticamente.")
    print("Pressione Ctrl+C para interromper a qualquer momento.\n")

    try:
        melhor_params, melhor_valor = optimizer.otimizar(
            params_iniciais=params_iniciais,
            tipos=tipos,
            tamanho_populacao=tamanho_populacao,
            num_geracoes=num_geracoes,
            taxa_crossover=taxa_crossover,
            taxa_mutacao=taxa_mutacao,
            limite_min=limite_min,
            limite_max=limite_max
        )

        optimizer.salvar_relatorio()

        print("\n" + "-"*70)
        print("COMANDO PARA REPRODUZIR O MELHOR RESULTADO:")
        params_str = " ".join(str(p) for p in melhor_params)
        print(f"\n{comando_modelo} {params_str}")
        print("\n" + "-"*70 + "\n")

        return True

    except KeyboardInterrupt:
        print("\n\n[AVISO] Otimizacao interrompida pelo usuario.")
        if optimizer.melhor_params:
            print(f"\nMelhor resultado ate agora:")
            print(f"Valor: {optimizer.melhor_valor:.6f}")
            print(f"Parametros: {optimizer.melhor_params}")
        return False

    except Exception as e:
        print(f"\n[ERRO] Erro durante otimizacao: {e}")
        import traceback
        traceback.print_exc()
        return False


def executar_particle_swarm():
    """Executa o Particle Swarm com configuracao automatica"""

    print("\n" + "="*70)
    print("CONFIGURACAO DO PARTICLE SWARM OPTIMIZATION")
    print("="*70 + "\n")

    # 1. Entrada completa: comando + padrao
    print("1. Digite o COMANDO + PADRAO (tudo em uma linha):")
    print("   Exemplos:")
    print("   - modelo10.exe baixo 1 1 1 1 1 1 1 1 1")
    print("   - ./modelo10.exe baixo 1 1.5 true 1")
    print("   - meu_modelo.exe 1 1 1 1 1 1 1 1")

    entrada_completa = input("\n   Digite: ").strip()

    if not entrada_completa:
        entrada_completa = "modelo10.exe baixo 1 1 1 1 1 1 1 1 1"
        print(f"   [PADRAO] Usando: {entrada_completa}")

    partes = entrada_completa.split()
    if len(partes) < 2:
        print("\n   [ERRO] Formato invalido! Use: comando param1 param2 ...")
        return False

    comando_modelo = partes[0]
    padrao = " ".join(partes[1:])

    print(f"\n   [OK] Comando detectado: {comando_modelo}")
    print(f"   [OK] Padrao detectado: {padrao}")

    # Criar otimizador temporario para detectar padrao
    optimizer_temp = ParticleSwarm(comando_modelo)
    params_iniciais, tipos = optimizer_temp.detectar_padrao(padrao)

    print(f"\n   [OK] Analise do padrao:")
    print(f"   - Quantidade de variaveis: {len(params_iniciais)}")
    print(f"   - Tipos: {tipos}")
    print(f"   - Valores iniciais: {params_iniciais}")

    # 2. Escolher modo: maximizar ou minimizar
    print("\n2. Modo de otimizacao:")
    print("   [1] Maximizar (buscar maior valor)")
    print("   [2] Minimizar (buscar menor valor)")
    modo_input = input("   Escolha [1]: ").strip()
    modo = 'minimizar' if modo_input == '2' else 'maximizar'
    print(f"   [OK] Modo selecionado: {modo.upper()}")

    # Criar otimizador com o modo escolhido
    optimizer = ParticleSwarm(comando_modelo, modo=modo)

    # 3. Configuracoes de otimizacao
    print("\n3. Configuracoes de otimizacao (pressione ENTER para usar padrao):")

    try:
        num_part_input = input("   Numero de particulas [30]: ").strip()
        num_particulas = int(num_part_input) if num_part_input else 30

        num_iter_input = input("   Numero de iteracoes [100]: ").strip()
        num_iteracoes = int(num_iter_input) if num_iter_input else 100

        w_input = input("   Peso de inercia (w) [0.7]: ").strip()
        w = float(w_input) if w_input else 0.7

        c1_input = input("   Constante cognitiva (c1) [1.5]: ").strip()
        c1 = float(c1_input) if c1_input else 1.5

        c2_input = input("   Constante social (c2) [1.5]: ").strip()
        c2 = float(c2_input) if c2_input else 1.5

        limite_min_input = input("   Limite minimo [1]: ").strip()
        limite_min = float(limite_min_input) if limite_min_input else 1

        limite_max_input = input("   Limite maximo [100]: ").strip()
        limite_max = float(limite_max_input) if limite_max_input else 100

    except ValueError:
        print("   [AVISO] Valor invalido. Usando configuracoes padrao.")
        num_particulas = 30
        num_iteracoes = 100
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        limite_min = 1
        limite_max = 100

    # 4. Executa otimizacao
    print("\n" + "="*70)
    print("INICIANDO OTIMIZACAO AUTOMATICA...")
    print("="*70)
    objetivo = "MAIOR VALOR" if modo == 'maximizar' else "MENOR VALOR"
    print(f"\nO algoritmo tentara encontrar o {objetivo} automaticamente.")
    print("Pressione Ctrl+C para interromper a qualquer momento.\n")

    try:
        melhor_params, melhor_valor = optimizer.otimizar(
            params_iniciais=params_iniciais,
            tipos=tipos,
            num_particulas=num_particulas,
            num_iteracoes=num_iteracoes,
            w=w,
            c1=c1,
            c2=c2,
            limite_min=limite_min,
            limite_max=limite_max
        )

        optimizer.salvar_relatorio()

        print("\n" + "-"*70)
        print("COMANDO PARA REPRODUZIR O MELHOR RESULTADO:")
        params_str = " ".join(str(p) for p in melhor_params)
        print(f"\n{comando_modelo} {params_str}")
        print("\n" + "-"*70 + "\n")

        return True

    except KeyboardInterrupt:
        print("\n\n[AVISO] Otimizacao interrompida pelo usuario.")
        if optimizer.melhor_params:
            print(f"\nMelhor resultado ate agora:")
            print(f"Valor: {optimizer.melhor_valor:.6f}")
            print(f"Parametros: {optimizer.melhor_params}")
        return False

    except Exception as e:
        print(f"\n[ERRO] Erro durante otimizacao: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Funcao principal"""

    while True:
        mostrar_menu()

        try:
            escolha = input("\nEscolha uma opcao: ").strip()

            if escolha == '0':
                print("\n[OK] Encerrando sistema. Ate logo!")
                sys.exit(0)

            elif escolha == '1':
                executar_pattern_search()
                input("\nPressione ENTER para voltar ao menu...")

            elif escolha == '2':
                executar_genetic_algorithm()
                input("\nPressione ENTER para voltar ao menu...")

            elif escolha == '3':
                executar_particle_swarm()
                input("\nPressione ENTER para voltar ao menu...")

            elif escolha == '4':
                print("\n[AVISO] Este metodo ainda nao foi implementado.")
                print("        Em breve estara disponivel!")
                input("\nPressione ENTER para voltar ao menu...")

            else:
                print("\n[ERRO] Opcao invalida! Escolha entre 0-4.")
                input("\nPressione ENTER para tentar novamente...")

        except KeyboardInterrupt:
            print("\n\n[OK] Encerrando sistema. Ate logo!")
            sys.exit(0)

        except Exception as e:
            print(f"\n[ERRO] Erro inesperado: {e}")
            import traceback
            traceback.print_exc()
            input("\nPressione ENTER para continuar...")


if __name__ == "__main__":
    main()
