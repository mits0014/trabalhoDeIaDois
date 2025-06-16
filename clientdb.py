import psycopg2
import json


def conectar(host, dbname, user, password, port): 
     try:
        # Conectar ao banco Supabase (PostgreSQL)
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        return conn
     except Exception as e:
        print("Erro ao conectar ao banco de dados:", e)
        return None


def salvar_resultado_modelo(nome, matriz_confusao, acuracia, precisao, recall, f1_score, parametros):
    """
    Salva os resultados do modelo em sua tabela específica no banco de dados.
    
    Args:
        nome (str): Nome do modelo, usado para determinar a tabela.
        matriz_confusao (list): Matriz de confusão.
        acuracia (float): Acurácia do modelo.
        precisao (float): Precisão média.
        recall (float): Recall médio.
        f1_score (float): F1-score médio.
        parametros (dict): Dicionário com os hiperparâmetros do modelo.
    """
    user="postgres.pbrgokfauzknintkqxgk" 
    password="9FwYA4-kG8mUemj"
    host="aws-0-sa-east-1.pooler.supabase.com"
    port=6543
    dbname="postgres"

    # Exemplo usando SQLite — substitua pela conexão do seu banco
    conn = conectar(host, dbname, user, password, port)
    cursor = conn.cursor()

    # Nome da tabela baseado no nome do modelo

    # Serializar matriz de confusão e parâmetros como JSON
    matriz_confusao_json = json.dumps(matriz_confusao)
    parametros_json = json.dumps(parametros)

    # Montar o comando SQL (cada tabela tem a mesma estrutura base)
    sql = """
    INSERT INTO resultados_modelos (
        modelo,
        acuracia,
        precisao,
        recall,
        f1_score,
        matriz_confusao,
        parametros
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        # Executar o comando
        cursor.execute(sql, (
            nome,
            acuracia,
            precisao,
            recall,
            f1_score,
            matriz_confusao_json,
            parametros_json
        ))
        conn.commit()
    except Exception as e:
        print("Erro ao salvar resultado do modelo:", e)
    finally:
        print("Resultado do modelo salvo com sucesso!")
        conn.close()




def conectar_e_inserir(
    host, dbname, user, password, port,
    parametros, acuracia, matriz_confusao
):
    try:
        # Conectar ao banco Supabase (PostgreSQL)
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        cur = conn.cursor()

        # Query de inserção
        insert_query = """
        INSERT INTO treinamento_modelo (
            parametros, acuracia, matriz_confusao
        ) VALUES (%s, %s, %s);
        """

        # Executar a query
        cur.execute(insert_query, (
            json.dumps(parametros),
            float(acuracia),
            json.dumps(matriz_confusao)
        ))

        # Salvar alterações e fechar conexão
        conn.commit()
        cur.close()
        conn.close()
        print("Dados inseridos com sucesso!")

    except Exception as e:
        print("Erro ao conectar ou inserir dados:", e)

def conectar_e_ler(
        host, dbname, user, password, port
):
    try:
        # Conectar ao banco Supabase (PostgreSQL)
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        cur = conn.cursor()

        # Query de leitura
        select_query = "SELECT * FROM treinamento_modelo;"
        cur.execute(select_query)

        # Obter os resultados
        rows = cur.fetchall()
        lista_banco = trata_banco(rows)

        # Fechar conexão
        cur.close()
        conn.close()

        return lista_banco

    except Exception as e:
        print("Erro ao conectar ou ler dados:", e)

def conectar_e_ler_teste(
        host, dbname, user, password, port
):
    try:
        # Conectar ao banco Supabase (PostgreSQL)
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        cur = conn.cursor()
        

        # Query de leitura
        select_query = "SELECT * FROM treinamento_modelo WHERE id = 25;"
        cur.execute(select_query)

        # Obter os resultados
        rows = cur.fetchall()
        lista_banco = trata_banco(rows)

        # Fechar conexão
        cur.close()
        conn.close()

        return lista_banco

    except Exception as e:
        print("Erro ao conectar ou ler dados:", e)

def trata_banco(rows):
    """
    Recebe uma lista de tuplas (linhas do banco) e retorna uma lista de dicionários,
    cada um representando uma linha com colunas nomeadas.
    """
    resultado = []
    for row in rows:
        obj = {
            "id": row[0],
            "data": row[1],
            "parametros": dict(row[2]),
            "acuracia": row[3],
            "matriz_confusao": row[4]
        }
        resultado.append(obj)
    return resultado