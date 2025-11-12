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

    def __init__(self, comando_modelo: str):
        """
        Inicializa o Pattern Search

        Args:
            comando_modelo: Comando do executavel (ex: 'modelo10.exe')
        """
        self.comando_modelo = comando_modelo
        self.melhor_valor = float('-inf')
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
            if valor > self.melhor_valor:
                self.melhor_valor = valor
                self.melhor_params = params.copy()
                print(f"[OK] Novo maximo: {valor:.6f} com {params}")

            # Salva no historico
            self.historico.append((params.copy(), valor))

            return valor

        except Exception as e:
            print(f"[ERRO] Erro ao executar modelo: {e}")
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
                 limite_min: Union[int, float] = 0,
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
                print("> Nenhum vizinho valido. Reduzindo step.")
                step_size = step_size * fator_reducao
                if 'int' in tipos:
                    step_size = max(step_minimo, int(step_size))
                continue

            # Avalia vizinhos
            melhor_vizinho = None
            melhor_valor_vizinho = valor_atual

            for viz in vizinhos:
                valor_viz = self.executar_modelo(viz)
                if valor_viz > melhor_valor_vizinho:
                    melhor_valor_vizinho = valor_viz
                    melhor_vizinho = viz

            # Se encontrou um vizinho melhor, move para ele
            if melhor_vizinho is not None:
                params_atual = melhor_vizinho
                valor_atual = melhor_valor_vizinho
                print(f"> Movendo para vizinho melhor: {params_atual} = {valor_atual:.6f}")
            else:
                # Nao encontrou melhoria, reduz o step
                step_size = step_size * fator_reducao
                if 'int' in tipos:
                    step_size = max(step_minimo, int(step_size))
                print(f"> Nenhuma melhoria. Reduzindo step para: {step_size}")

        # Resultado final
        print("\n" + "="*70)
        print("OTIMIZACAO CONCLUIDA")
        print("="*70)
        print(f"Total de execucoes: {self.total_execucoes}")
        print(f"Iteracoes realizadas: {iteracao}")
        print(f"Valor maximo encontrado: {self.melhor_valor:.6f}")
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


def mostrar_menu():
    """Mostra o menu principal"""
    print("\n" + "="*70)
    print("        SISTEMA DE OTIMIZACAO AUTOMATICA")
    print("="*70)
    print("\nMETODOS DE OTIMIZACAO DISPONIVEIS:\n")
    print("  [1] Pattern Search (Busca por Padroes)")
    print("  [2] Simulated Annealing (Em breve...)")
    print("  [3] Genetic Algorithm (Em breve...)")
    print("  [4] Particle Swarm (Em breve...)")
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

    # Criar otimizador
    optimizer = PatternSearch(comando_modelo)

    # Detectar padrao
    params_iniciais, tipos = optimizer.detectar_padrao(padrao)

    print(f"\n   [OK] Analise do padrao:")
    print(f"   - Quantidade de variaveis: {len(params_iniciais)}")
    print(f"   - Tipos: {tipos}")
    print(f"   - Valores iniciais: {params_iniciais}")

    # 2. Configuracoes de otimizacao
    print("\n2. Configuracoes de otimizacao (pressione ENTER para usar padrao):")

    try:
        step_inicial_input = input("   Step inicial [20]: ").strip()
        step_inicial = float(step_inicial_input) if step_inicial_input else 20

        step_minimo_input = input("   Step minimo [1]: ").strip()
        step_minimo = float(step_minimo_input) if step_minimo_input else 1

        max_iter_input = input("   Maximo de iteracoes [1000]: ").strip()
        max_iter = int(max_iter_input) if max_iter_input else 1000

        limite_min_input = input("   Limite minimo [0]: ").strip()
        limite_min = float(limite_min_input) if limite_min_input else 0

        limite_max_input = input("   Limite maximo [100]: ").strip()
        limite_max = float(limite_max_input) if limite_max_input else 100

    except ValueError:
        print("   [AVISO] Valor invalido. Usando configuracoes padrao.")
        step_inicial = 20
        step_minimo = 1
        max_iter = 1000
        limite_min = 0
        limite_max = 100

    # 3. Executa otimizacao
    print("\n" + "="*70)
    print("INICIANDO OTIMIZACAO AUTOMATICA...")
    print("="*70)
    print("\nO algoritmo tentara encontrar o MAIOR VALOR automaticamente.")
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

            elif escolha in ['2', '3', '4']:
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
