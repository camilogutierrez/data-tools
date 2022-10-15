# from IPython import display
from colorama import Back, Style
from datetime import date
from django import db
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from typing import List
from typing import Type
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import unicodedata
from pathlib import Path
try:
    from django.core.validators import validate_email
    from django.db.models import Count, Q, F, Value
    from django.db.models.query import QuerySet
    from django.contrib.postgres.aggregates.general import ArrayAgg
    from django.core.exceptions import MultipleObjectsReturned
except:
    pass
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from core.models import *
    from listas.models import *
    from django.db import connection
    db_name = connection.settings_dict
    print('USANDO BASE DE DATOS EN PUERTO:', db_name['HOST'])
except:
    pass
pd.set_option('display.max_columns', None)
parentesis = re.compile("\((.*?)\)")
REGEX_PARENTESIS = re.compile(r'\s*\(([^)]*)\)\s*')
REGEX_ID = re.compile('\d{5,11}')
REGEX_CELULAR = re.compile('\d{10}')
REGEX_EMAIL = re.compile('[\w\d\.-]+@[\w\.-]+', re.I)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

COLORES = ['#ed2a6e', '#4465b9']

PATH_CIFRAS = Path(
    r'C:\Users\Camilo\Proyectos\MINTIC\sistema_informacion_mintic\Requerimientos\Cifras')

PATH_DATOS = Path(r"C:\Users\Camilo\Proyectos\MINTIC\Excel SI\datos")

def get_path(path):
    relativo = os.path.relpath(os.getcwd(), 'C:\\Users\\Camilo\\Proyectos\\MINTIC\\sistema_informacion_mintic\\datos')
    return PATH_DATOS/relativo/path

def get_inside_parentesis(df, return_listas=False):
    """[summary]

    Args:
        df ([type]): [description]
        return_listas (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    df_categorias = get_categories(df)
    columnas = df_categorias.loc[df_categorias.unicos < 50].index
    columnas_con_parentesis = []
    for category in columnas:
        if (df.loc[df[category].notna(), category].astype(str).str.contains(REGEX_PARENTESIS, na="(nan)").all()):
            print(category)
            if return_listas:
                df[category] = df[category].astype(
                    str).str.findall(REGEX_PARENTESIS)
            else:
                df[category] = df[category].astype(
                    str).str.extract(REGEX_PARENTESIS)
            columnas_con_parentesis.append(category)

            # TODO mejorarlo con esto
            # f_publico.que_lenguaje_de_programacion_manejan.str.findall(parentesis)

    return columnas_con_parentesis


def reemplazar_string(text):
    reemplazar = {
        "enero": "01",
        "febrero": "02",
        "marzo": "03",
        "abril": "04",
        "mayo": "05",
        "junio": "06",
        "julio": "07",
        "agosto": "08",
        "septiembre": "09",
        "octubre": "10",
        "noviembre": "11",
        "diciembre": "12",
    }
    return "-".join(
        [reemplazar.get(word, word) for word in re.split("\W", text)])


# def c():
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
# SMALL_SIZE = 12
# MEDIUM_SIZE = 12
# BIGGER_SIZE = 14

# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
####  Muy Importante   ######
plt.rcParams.keys()  # Me dice los parametros como estan en MAtplotlib #Muy IMportante

# plt.rcParams["font.family"] = "Cambria"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=[5, 3])


def agregar_db_list_items(lista_items, codigo_lista_tipo, database='default'):
    """[summary]

    Args:
        lista_items ([type]): [description]
        codigo_lista_tipo ([type]): [description]
        database (str, optional): [description]. Defaults to 'default'.
    """
    lista_items = [estandarizar(item) for item in lista_items]
    for codigo in lista_items:
        list_type = ListType.objects.using(
            database).get(codigo=codigo_lista_tipo)
        e, created = ListItem.objects.using(database).get_or_create(codigo=codigo,
                                                                    list_type=list_type,
                                                                    defaults=dict(
                                                                        nombre=desestandarizar(
                                                                            codigo),
                                                                        description=desestandarizar(codigo)))
        if created:
            print(f'Codigo - {codigo} - creado')


def organizar_columnas(df):
    df.columns = df.columns.map(remove_accents)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = "{:.0f}".format(p.get_height())
            ax.text(_x, _y, value, ha="center", va="bottom")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def get_dummies_list(s):
    '''La serie s tiene q ser una columna de valores tipo listas'''
    '''https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies/29036042'''
    return pd.get_dummies(s.apply(pd.Series).stack()).groupby(level=0).sum()


def get_categories(df, limite=20):

    cols_disponibles = df.columns[~df.applymap(
        lambda x: isinstance(x, list)).all(axis=0)]
    columnas_descartadas = df.columns[~df.columns.isin(cols_disponibles)]
    print('Columnas multiselect:', columnas_descartadas.to_list())
    for col in columnas_descartadas:
        print(f'\nColumna {col} - Tipo Multiselct con opciones:')
        print(' | '.join(df[col].explode().dropna().unique()))
    df_unicos = df.apply(lambda s: s.explode().dropna().unique().shape).T.set_axis(
        ['unicos'], axis=1)
    df_categories = df.apply(lambda x: " | ".join(map(
        str, x.explode().dropna().unique())) if x.explode().dropna().unique().shape[0] < limite else '-', axis=0).to_frame('categorias')
    df_faltantes = df.apply(
        lambda s: s.isnull()).sum().to_frame('faltantes')
    df_conteo = df.count().to_frame('num_datos')
    df_tipo = df.dtypes.to_frame('tipo')
    df_pct = (df.isnull().mean()*100).round(1).to_frame('% faltante')
    df_len = df.apply(lambda x: x.astype('string').str.len()
                      ).max(axis=0).to_frame('long_max')
    return pd.concat([df_categories, df_unicos, df_conteo, df_faltantes, df_pct, df_tipo, df_len], axis=1).sort_index()


def modo_full(full=1):
    if full == 1:
        pd.set_option('display.max_rows', 250)
        pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_colwidth', None)
    else:
        pd.set_option('display.max_rows', 100, silent=True)

        # pd.reset_option('^display.', silent=True)


def nombres_apellidos(nombre_completo: str):
    """df[['nombres','apellidos']] = df.Nombre.apply(nombres_apellidos).tolist()
    SI EMPIEZA EN APELLIDO Y DESPUES NOMBRE USAR LA FUNCION "apellidos_nombres"
    """
    nombre_completo = re.sub(
        ' +', ' ', nombre_completo)  # Quitar espacios duplicados
    nombre_completo = nombre_completo.strip().split(" ")
    if len(nombre_completo) > 0:
        if len(nombre_completo) == 4:
            nombres = nombre_completo[:2]
            apellidos = nombre_completo[2:]
            return ' '.join(nombres), ' '.join(apellidos)
        elif len(nombre_completo) == 3:
            nombres = nombre_completo[:1]
            apellidos = nombre_completo[1:]
            return ' '.join(nombres), ' '.join(apellidos)
        elif len(nombre_completo) == 2:
            nombres = nombre_completo[0]
            apellidos = nombre_completo[1]
            return nombres, apellidos
        elif len(nombre_completo) > 4:
            apellidos = nombre_completo[-2:]
            nombres = nombre_completo[:-2]
            return ' '.join(nombres), ' '.join(apellidos)
        else:
            return nombre_completo[0], None
    else:
        return None, None


def apellidos_nombres(nombre_completo: str):
    """df[['nombres','apellidos']] = df.Nombre.apply(apellidos_nombres).tolist()
    """
    nombre_completo = re.sub(
        ' +', ' ', nombre_completo)  # Quitar espacios duplicados
    nombre_completo = nombre_completo.strip().split(" ")
    if len(nombre_completo) > 0:
        if len(nombre_completo) == 4:
            apellidos = nombre_completo[:2]
            nombres = nombre_completo[2:]
            return ' '.join(nombres), ' '.join(apellidos)
        elif len(nombre_completo) == 3:
            nombres = nombre_completo[-1]
            apellidos = nombre_completo[:-1]
            return nombres, ' '.join(apellidos)
        elif len(nombre_completo) == 2:
            nombres = nombre_completo[0]
            apellidos = nombre_completo[1]
            return nombres, apellidos
        elif len(nombre_completo) > 4:
            apellidos = nombre_completo[:2]
            nombres = nombre_completo[2:]
            return ' '.join(nombres), ' '.join(apellidos)
        else:
            return nombre_completo[0], None
    else:
        return None, None


def full_print(df):
    from IPython.display import display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    display(df)
    pd.reset_option('display.max_rows')
    # pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


fp = full_print


def remove_accents(input_str, conservar_ñ=True) -> str:
    """
    Remove accents from a string.
    """
    def normalizar(input_str):

        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        text = only_ascii.decode("utf-8")
        return text

    if isinstance(input_str, str):
        if conservar_ñ:
            # good_accents = {
            #     u'\N{COMBINING TILDE}',
            #     u'\N{COMBINING CEDILLA}'
            # }
            # return ''.join(c for c in unicodedata.normalize('NFKC', input_str)
            #                if (unicodedata.category(c) != 'Mn')
            #                or c in good_accents)
            return ''.join([normalizar(l) if l.upper() not in ['Ñ', "¿", '¡'] else l for l in input_str])

        else:
            text = normalizar(input_str)
            return text
    else:
        input_str

        # TODO
        #         good_accents = {
        #     u'\N{COMBINING TILDE}',
        #     u'\N{COMBINING CEDILLA}'
        # }
        # ''.join(c for c in unicodedata.normalize('NFKD', 'ñ')
        #                        if (unicodedata.category(c) != 'Mn')
        #                            or c in good_accents)


def find(df, **kwargs):
    """[Encontrar las estaciones]

    Ejemplo: find_station(nombre='Mapi', categoria = 'limn')
        si hay espacios en el nombre de la columna: find_station(**{'CON ESPACIOS':'limni'})
    Si hay varias que cumplen, siempre lee la primera
    Args:
        read (bool, optional): [description]. Defaults to False.
    """
    df_booleans = {}
    for key in kwargs.keys():
        df_booleans[key] = df[key].astype('string').str.contains(
            kwargs.get(key), case=False, na=False)
    # Se determina las filas que no continen ningun valor de la KEY
    df_posiciones = pd.DataFrame(df_booleans).all(axis=1, skipna=False)
    df_posiciones[df_posiciones.isna()] = False
    return df.loc[df_posiciones]


def calculate_age(born, fecha = None):
    try:
        born = pd.to_datetime(born)
        if fecha is None:
            fecha = pd.to_datetime('today')
        return (fecha - born).astype('timedelta64[Y]')
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except:
        return None


def check_difference(lista1, lista2, estandarizado=False):
    if estandarizado:
        lista1 = [estandarizar(x) for x in lista1]
        lista2 = [estandarizar(x) for x in lista2]
    '''Determina que hay en la lista 1 que no hay en la dos'''
    return pd.Series(lista1)[~np.isin(lista1, lista2)].reset_index(drop=True)


def comparar_columnas(*listas_df: List[pd.DataFrame], return_truecols=None, filtrar_regex=None, col_names=None) -> Type[pd.DataFrame]:
    """Si se necesita el dataframe con el atributo X.data se accede.

    Returns:
        [type]: [description]
    """
    lista_df = list(listas_df)
    for df in lista_df:
        df.columns = df.columns.astype('string')
        if filtrar_regex is not None:
            # TODO
            df = df.filter(regex=filtrar_regex)

    print(type(lista_df))
    columnas = list(itertools.chain.from_iterable(
        [df.columns.to_list() for df in lista_df]))
    columnas = list(set(columnas))
    contiene_columna = [{col: df.columns.isin(
        [col]).any() for col in columnas} for df in lista_df]
    df_comparacion = pd.DataFrame(contiene_columna).T.sort_index()
    if col_names:
        df_comparacion.columns = col_names
    if return_truecols is True:
        return df_comparacion.loc[df_comparacion.all(axis=1)]
    return df_comparacion.sort_index().style.applymap(lambda x: 'color: green' if x else 'color: red')


def estandarizar(x):
    if isinstance(x, str):
        return remove_accents(x).strip().replace(" ", "_").lower().strip()
    else:
        return x


def desestandarizar(x):
    if isinstance(x, str):
        return x.strip().replace("_", " ").capitalize()
    else:
        return x


def crear_preguntas(convocatoria, codigo=None, nombre=None, abierta=True):
    # convocatoria = Convocatoria.objects.get(
    #     nombre='Convocatoria ruta 2 - Misión Tic 2022')
    tipo_preg = ListType.objects.get(codigo='tipo_preguntas').listitem_set.get(
        codigo='abiertas' if abierta else 'seleccion')
    p, created = Pregunta.objects.get_or_create(codigo=codigo,
                                                defaults=dict(
                                                    nombre=nombre,
                                                    tipo=tipo_preg,
                                                    estado=ListType.objects.get(codigo='estados').listitem_set.get(codigo='activo')))
    if created:
        print('Pregunta creada')

    else:
        print('Ya existia')
    convocatoria.preguntas.add(p)


def poblar_opciones(col_pregunta, df, Pregunta):
    opciones_dict = pd.Series(df[col_pregunta].unique())
    opciones_dict = opciones_dict[~opciones_dict.isna()]
    df.loc[~df[col_pregunta].isna(), col_pregunta] = df.loc[~df[col_pregunta].isna(
    ), col_pregunta].apply(estandarizar)
    opciones_dict_list = [{'codigo': estandarizar(
        lab), 'nombre': lab} for lab in opciones_dict]
#     print(opciones_dict)
    preg, _ = Pregunta.objects.get_or_create(
        codigo=col_pregunta)  # .opciones.add(Opa)

    # preg.opciones.all().delete()
    for kwgs in opciones_dict_list:
        preg.opciones.create(**kwgs)

    opciones = [(p.nombre, p.codigo)
                for p in Pregunta.objects.get(codigo=col_pregunta).opciones.all()]
    print(opciones)


def agregar_opciones(df, codigo_pregunta=None, db='default'):
    if df[codigo_pregunta].apply(lambda x: isinstance(x, list)).all(axis=0):
        # TODO: SACAR VALORES UNICOS CUANDO LLEGA UNA LISTA [MULTISELECT]
        listado = df[codigo_pregunta].explode().unique()
    else:
        listado = df[codigo_pregunta].explode().dropna().unique()
    p = Pregunta.objects.using(db).get(codigo=codigo_pregunta)
    if len(listado) > 0:
        listado = listado[~pd.isnull(listado)]
        for item in listado:
            opcion, _ = Opcion.objects.using(db).get_or_create(
                codigo=estandarizar(item), defaults={'nombre': desestandarizar(item)})
            p.opciones.add(opcion)


def agregar_estados(estado, descripcion = None, query_resultado=None, llaves=None, convocatoria='ruta 1 - 2021 - Mision TIC', db_name='default'):
    db_name = query_resultado[0]._state.db

    if query_resultado is None:
        convocatoria = Convocatoria.objects.using(
            db_name).get(nombre__icontains=convocatoria)
        query_resultado = ResultadoAspiranteConvocatoria.objects.using(db_name).filter(convocatoria=convocatoria,
                                                                                       puesto_llegada__in=llaves)
    nuevo_estado = ListItem.objects.using(db_name).get(codigo=estado,
                                                       list_type__codigo='estados_usuarios')

    conteo_inicial = Estado.objects.using(db_name).all().count()

    Estado.objects.using(db_name).bulk_create([Estado(estado=nuevo_estado, proceso_aspirante=resultado,
                                                      origen_estado=descripcion) for resultado in query_resultado],
                                              ignore_conflicts=True)

    conteo_final = Estado.objects.using(db_name).all().count()
    print("Se agregaron --", conteo_final -
          conteo_inicial, "-- nuevos estados.")

def agregar_etiquetas(nombre, codigo = None, descripcion = None, query_resultado=None):
    db_name = query_resultado[0]._state.db
    print(db_name)
    tipo_etiqueta = ListType.objects.using(db_name).get(nombre='Etiquetas', codigo = 'etiquetas')
    if codigo is None:
        codigo = estandarizar(nombre)
    etiqueta, _ = ListItem.objects.using(db_name).get_or_create(nombre = nombre, codigo = codigo,
                                                 list_type = tipo_etiqueta)
    
    conteo_inicial = Tag.objects.using(db_name).all().count()

    Tag.objects.using(db_name).bulk_create([Tag(resultado_aspirante_tag = resultado, etiqueta = etiqueta) for resultado in query_resultado], 
                            ignore_conflicts = True)
    
    conteo_final = Tag.objects.using(db_name).all().count()
    print("Se agregaron --", conteo_final -
          conteo_inicial, "-- nuevas etiquetas.")

def agregar_preseleccionados(convocatoria, db_name='default'):
    query_preseleccionados = ResultadoAspiranteConvocatoria.objects.filter(convocatoria__nombre='Programación para Niños y Niñas - 2021-01',
                                                                           estado__nombre='Preseleccionado')
    no_funciona = []
    for preseleccionado in query_preseleccionados:
        try:
            preseleccionado.pk = None
            preseleccionado._state.adding = True
            preseleccionado.convocatoria = Convocatoria.objects.get(
                nombre='Programación para Niños y Niñas - 2021-02')
            preseleccionado.save()
        except:
            no_funciona.append(preseleccionado)


def actualizar_estados(estado_viejo: str, estado_nuevo: str,  descripcion: str, llaves=None, convocatoria=None, db_name='default', query_resultado=None):
    if query_resultado is None:
        convocatoria = Convocatoria.objects.using(
            db_name).get(nombre__icontains=convocatoria)
        query_resultado = ResultadoAspiranteConvocatoria.objects.using(db_name).filter(convocatoria=convocatoria,
                                                                                       puesto_llegada__in=llaves)

    for index, resultado in enumerate(query_resultado):
        print(index, end=', ')
        Estado.objects.using(db_name).filter(proceso_aspirante=resultado, estado__codigo=estado_viejo).update(
            estado=ListItem.objects.get(codigo=estado_nuevo), origen_estado=descripcion)


def eliminar_columnas_sin_datos(df):
    # COLUMNAS SIN NINGUN  DATO.
    col_no_data = df.columns[df.isna().all()]
    print('Columnas eliminadas:\n', col_no_data)

    df.drop(columns=col_no_data, inplace=True)

# * Saving dataframes


def save_df_fit(df_save, filename='output.xlsx', **kwargs):
    """
    Save a dataframe with auto-adjusted columns width.
    """
    df = df_save.copy()

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # for sheetname, df in dfs.items():  # loop through `dict` of dataframes
    df.to_excel(writer, **kwargs)  # send df to writer
    worksheet = writer.sheets["Sheet1"]
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype("string").str.len().max(),  # len of largest item
            len(str(col))  # len of column name/header
        )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.save()


def save_df_one_sheet(df_list, sheet_name, file_name, spaces):
    '''Put multiple dataframes into one xlsx sheet

    If you have a dict with keys as sheet names and values as dataframes you can use:
        keys, values = zip(*df_dict.items())
        save_df_multiple_sheets(values, keys, "cruce.xlsx")
    '''
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter',)
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheet_name,
                           startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()


def save_df_multiple_sheets(df_list, sheet_list, file_name):
    '''Put multiple dataframes across separate tabs/sheets'''
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet,
                           startrow=0, startcol=0, index=True)
    writer.save()


# -  Extraer las respuestas como JSON
def sacar_json(lista, codigo_pregunta):
    p = Pregunta.objects.get(codigo=codigo_pregunta)
    tipo = p.tipo.codigo
    if tipo == 'seleccion':
        lista = lista if isinstance(lista, list) else [lista]
        opciones = [estandarizar(opcion) if isinstance(opcion, str) else opcion
                    for opcion in lista]  # sacar_codigo_opciones
    else:
        opciones = lista

    return {'tipo': tipo, 'opciones': opciones}


def cargar_respuestas(row, preguntas_convocatoria):

    dict_respuestas = {}
    dict_preguntas = row[preguntas_convocatoria].to_dict()

    for key in dict_preguntas.keys():
        lista = dict_preguntas[key]
        codigo_pregunta = key
        dict_respuestas[key] = sacar_json(lista, codigo_pregunta)

    return dict_respuestas


def procesar_municipio_depto(df_procesar, municipio_referencia, departamento_referencia, with_accents=True):
    '''Retorna el df_procesar con dos columnas: codigo_municipio, codigo_dpto_modificado. Resetea los indices'''

    df_procesar.reset_index(drop=True, inplace=True)
    df = df_procesar.copy()

    if municipio_referencia != 'municipio':
        df.drop(columns=['municipio'], inplace=True, errors='ignore')
    df.rename(columns={municipio_referencia: 'municipio',
                       departamento_referencia: 'departamento'}, inplace=True)
    df.municipio = df.municipio.astype('string').str.upper()
    df.departamento = df.departamento.astype('string').str.upper()
    num_municipios_unicos = df[['municipio', 'departamento']].dropna(
        subset=['municipio']).astype(str).eval('municipio+departamento').unique().shape[0]
    print('num_municipios_unicos', num_municipios_unicos)

    if df_procesar.columns.str.contains('codigo_municipio|codigo_dpto_modificado', regex = True).any():
        print('No se proceso. Ya contiene los codigos Divipola')
        raise ValueError

    label = '_oficial' if with_accents else ''
    keys_departamento = ['nombre_dpto' + label, 'nombre_municipio' + label]
    # print(label, keys_departamento, df.head())

    if with_accents == False:
        df.loc[~df.departamento.isna(), 'departamento'] = df.loc[~df.departamento.isna(
        ), 'departamento'].astype(str).apply(remove_accents)
        df.loc[~df.municipio.isna(), 'municipio'] = df.loc[~df.municipio.isna(),
                                                           'municipio'].astype(str).apply(remove_accents)

    df_codigo_dpto = pd.merge(df[['departamento']], df_divipola[['nombre_dpto' + label, 'codigo_dpto_modificado']].drop_duplicates(),
                              how='left',
                              left_on='departamento',
                              right_on='nombre_dpto'+label).drop(columns=['departamento', 'nombre_dpto' + label])

    df_codigo_muni = pd.merge(df[['departamento', 'municipio']], df_divipola[keys_departamento + ['codigo_municipio']],
                              how='left',
                              left_on=['departamento', 'municipio'],
                              validate='m:1',
                              right_on=keys_departamento).drop(columns=keys_departamento + ['departamento', 'municipio'], axis=1)
    df_concatenado = pd.concat(
        [df_procesar, df_codigo_dpto, df_codigo_muni], axis=1,)
    df_concatenado[['codigo_dpto_modificado', 'codigo_municipio']] = df_concatenado[[
        'codigo_dpto_modificado', 'codigo_municipio']] .astype("Int64")
    return df_concatenado


def procesar_df(df, upper=False):
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda s: re.sub(
        '\s{2,}', ' ', s.strip().replace(u'\xa0', u' ')) if isinstance(s, str) else s)
    if upper:
        df = df.applymap(lambda s: s.upper()
                         if isinstance(s, str) else s).copy()

    def replace(match):
        word = match.group(1)
        if word not in ['de', 'del']:
            return word.title()
        return word
    try:
        # Pone en mayuscula la primera palabra excepto "de" y "del"
        df.nombres = df.nombres.str.replace(r'(\w+)', replace, regex=True)
        df.apellidos = df.apellidos.str.replace(r'(\w+)', replace, regex=True)
    except:
        pass
    df.columns = df.columns.str.replace('^desc_', '', regex=True)
    df = df.convert_dtypes()
    return df


def validar_correo(s):
    try:
        validate_email(s)
        return True
    except:
        return False


def fill_na_merge(df, columna):
    '''Llena los datos faltantes despues de un merge daldnole prioridad al LEFT'''
    '''Ls columnas con ese patron se pueden extraer asi: df.filter(regex = '_\w$').columns.str.rstrip('_xy').drop_duplicates()'''
    df_revisar = df.filter(regex=columna+'_\w$')
    if df_revisar.empty:
        raise ValueError('DataFrame vacio')
    if df_revisar.shape[1] > 2:
        raise ValueError(
            f'Las columnas con el nombre {columna} no es de longitud 2.')
    df_revisar = df_revisar.astype('string')
    # return df_revisar
    # Si son iguales la comparacion es empty
    if df_revisar.iloc[:, 0].compare(df_revisar.iloc[:, 1]).empty:
        df[columna] = df.pop(
            columna + '_x').astype('string').combine_first(df.pop(columna + '_y').astype('string'))
        print(columna, 'changed')
    else:
        return df_revisar.iloc[:, 0].compare(df_revisar.iloc[:, 1])


def get_na_columns(df, column):
    return df.loc[df[column].isna()]


def get_duplicated(df, column):
    return df.loc[df[column].duplicated(keep=False)].sort_values(column)


df_divipola = pd.read_pickle(os.path.join(PROJECT_ROOT, "divipola.pkl"))
df_divipola.nombre_municipio = df_divipola.nombre_municipio_oficial.apply(
    remove_accents)
df_divipola['nombre_municipio_sin_n'] = df_divipola['nombre_municipio'].apply(
    remove_accents, conservar_ñ=False)
assert df_divipola.nombre_municipio.nunique(
) == df_divipola.nombre_municipio_sin_n.nunique()
df_divipola.codigo_municipio = df_divipola.codigo_municipio.astype(int)


def extraer_fuzzy(municipio_to_eval, departamento, min_score=80):

    dict_municipios = df_divipola.groupby(
        'nombre_dpto')['nombre_municipio'].apply(list).to_dict()
#     scorer = fuzz.partial_token_sort_ratio
    scorer = fuzz.token_set_ratio
#     scorer=  fuzz.partial_token_sort_ratio

    if pd.notnull(municipio_to_eval) and dict_municipios.get(departamento):
        best_value = process.extractOne(
            municipio_to_eval, dict_municipios[departamento], scorer=scorer, score_cutoff=min_score)
        return best_value[0] if best_value else pd.NA
    else:
        return pd.NA


def get_municipios_malos(df, col_municipio='municipio', col_dpto='departamento', estandarizado=False):
    # no_presente = get_municipios_malos(df_icetex, 'municipio')
    '''USAR iloc'''
    sufijo = '_oficial' if estandarizado is True else ''
    no_presente = pd.merge(df, df_divipola,
                           left_on=[col_municipio, col_dpto],
                           right_on=['nombre_municipio' +
                                     sufijo, 'nombre_dpto'+sufijo],
                           how='left', validate='m:1',
                           indicator=True).query('_merge == "left_only"').drop(['_merge'], axis=1)
    return no_presente.index


def corregir_muni_fuzzy(df_procesar, reemplazar=False, col_municipio='municipio', col_dpto='departamento'):
    df = df_procesar[[col_dpto, col_municipio]].reset_index().copy()

    no_presente = get_municipios_malos(df, col_municipio, col_dpto)
    df['municipio_estandar'] = df[col_municipio].mask(
        df.index.isin(no_presente))

    df['fuzzy'] = df.loc[df['municipio_estandar'].isna()].apply(
        lambda row: extraer_fuzzy(row.get(col_municipio), row.get(col_dpto)), axis=1)

    if reemplazar is True:
        # df['municipio_estandar'] = df['municipio_estandar'].fillna(df['fuzzy'])
        return df.set_index('index')['municipio_estandar'].fillna(df.set_index('index')['fuzzy'])

    revisar = df.loc[df['municipio_estandar'].isna(), [col_dpto, col_municipio, 'fuzzy']].sort_values(
        col_municipio).drop_duplicates()  # .pipe(full_print)
    revisar.pipe(full_print)
    return revisar


def verificar_divipola(df, col_municipio='municipio', col_dpto='departamento', **kwargs):
    """Retorna la diferencia de muni, y la de los dptos.
    """

    no_presente = get_municipios_malos(
        df, col_municipio, col_dpto, **kwargs)
    # MUNICIPIOS
    df1 = df.iloc[no_presente][[col_dpto, col_municipio]
                               ].dropna().value_counts().sort_index(level=1)
    # DEPARTAMENTOS
    df2 = df.loc[~df[col_dpto].isin(df_divipola.nombre_dpto), [
        col_municipio, col_dpto]].value_counts()
    return df1, df2


def get_df_capitales(accents=True):
    path_capitales = r"C:\Users\Camilo\Proyectos\MINTIC\sistema_informacion_mintic\datos\capitales.TXT"
    path_capitales = os.path.join(PROJECT_ROOT, "capitales.TXT")
    df_capitales = pd.read_csv(path_capitales, sep=':', header=None)
    df_capitales.columns = ['departamento', 'capital_departamento']
    df_capitales = df_capitales.applymap(lambda s: s.upper().strip())
    if not accents:
        df_capitales.departamento = df_capitales.departamento.apply(
            remove_accents)
        df_capitales.capital_departamento = df_capitales.capital_departamento.apply(
            remove_accents)
    return df_capitales


def get_colegio():
    df_colegios = get_pickle(
        r"C:\Users\Camilo\Proyectos\MINTIC\Excel SI\datos\ColegiosEstablecimientosSINEB.xlsx",)
    organizar_columnas(df_colegios)
    df_colegios.codigo = df_colegios.codigo.astype('Int64')

    df_colegios = df_colegios[['secretaria', 'codigo_departamento', 'departamento', 'codigo_municipio',
                               'municipio', 'codigo', 'nombre', 'direccion', 'tipo_establecimiento', 'sector', 'genero',
                              'zona']]
    df_colegios = df_colegios.add_prefix('colegio_')
    return df_colegios


def get_mapeo_divipola():
    mapeo_departamentos_divipola = df_divipola.set_index('nombre_dpto')[
        ['nombre_dpto_oficial']].drop_duplicates().to_dict()['nombre_dpto_oficial']
    mapeo_municipios_divipola = df_divipola.set_index('nombre_municipio')[
        ['nombre_municipio_oficial']].drop_duplicates().to_dict()['nombre_municipio_oficial']
    mapeo_municipios_n = df_divipola.set_index('nombre_municipio_sin_n')[
        ['nombre_municipio_oficial']].drop_duplicates().to_dict()['nombre_municipio_oficial']

    # Diccionario llave: codigo divipola en string
    df_municipio_codigo = df_divipola[[
        'codigo_municipio', 'nombre_municipio']].copy()
    df_municipio_codigo.codigo_municipio = df_municipio_codigo.codigo_municipio.astype(
        'float').astype('Int64').astype('string')
    mapeo_codigo_municipio = df_municipio_codigo.set_index('codigo_municipio')[
        'nombre_municipio']
    return mapeo_departamentos_divipola, mapeo_municipios_divipola, mapeo_municipios_n, mapeo_codigo_municipio


class OrganizarDivipola:
    """
    Secuencia:

    1.  organizar_divipola = OrganizarDivipola(df_organizar, 'Municipio residencia empleado', 'Departamento residencia empleado')
    2.  errores_muni, errores_dpto = organizar_divipola.verificar_divipola()
    --Opcional:
        Rellenar con capitales: organizar_divipola.fill_capitales()
    organizar_divipola.corregir_muni_fuzzy()    
        Si esta de acuerdo con la columna fuzzy: organizar_divipola.corregir_muni_fuzzy(inplace  =True)
    organizar_divipola.organizar_tildes()
    organizar_divipola.organizar_divipola().to_clipboard(index = False)

    Finalmente:
    df[['codigo_dpto_modificado','codigo_municipio']] = organizar_divipola.organizar_divipola(oficial = False)[['codigo_dpto_modificado','codigo_municipio']]
    """

    def __init__(self, df, columna_municipio, col_dpto):
        self.col_municipio = columna_municipio
        self.col_dpto = col_dpto
        self.df = df[[self.col_dpto, self.col_municipio]].copy()
        self.df = procesar_df(self.df)
        if not df.index.is_monotonic_increasing or not df.index.is_unique or not(df.index.to_series().diff().max() == 1):
            raise ValueError('Indice debe ser creciente y unico.')
        self.procesar_columnas()
        self.organizar()

    def get_mapeo(self):
        mapeo_departamentos_divipola, mapeo_municipios_divipola, mapeo_municipios_n, mapeo_codigo_municipio = get_mapeo_divipola()

        # Guardarlo en atributos.
        self.mapeo_dpto = mapeo_departamentos_divipola
        self.mapeo_muni = mapeo_municipios_divipola
        self.mapeo_muni_n = mapeo_municipios_n
        self.mapeo_muni_codigo = mapeo_codigo_municipio

    def organizar_n(self):
        self.get_mapeo()
        self.df[self.col_municipio].replace(self.mapeo_muni_n, inplace=True)

    def organizar_tildes(self):
        self.get_mapeo()
        self.df[self.col_dpto].replace(self.mapeo_dpto, inplace=True)
        self.df[self.col_municipio].replace(self.mapeo_muni, inplace=True)
        self.df[self.col_municipio].replace(
            self.mapeo_muni_codigo, inplace=True)

    def quitar_tildes(self):
        self.df = self.df.applymap(lambda s: remove_accents(s))

    def procesar_columnas(self):
        self.df[self.col_dpto] = self.df[self.col_dpto].astype(
            'string').str.upper().apply(remove_accents)
        self.df[self.col_municipio] = self.df[self.col_municipio].astype(
            'string').str.upper().apply(remove_accents)

    def organizar(self):
        mapeo_municipios = {
                            '.*MOMPOS.*': 'SANTA CRUZ DE MOMPOX',
                            '.*BOGOTA.*': 'BOGOTA, D.C.',
                            '.*CUCUTA.*': 'SAN JOSE DE CUCUTA'
                            }
        mapeo_dptos = {'.*BOGOTA.*': 'BOGOTA, D.C.',
                       '.*VALLE.*': 'VALLE DEL CAUCA',
                       '.*SAN ANDR.*': 'ARCHIPIELAGO DE SAN ANDRES, PROVIDENCIA Y SANTA CATALINA',
                       '.*NARINO.*': 'NARIÑO',
                       '.*GUAJIRA.*': 'LA GUAJIRA',
                       }

        self.df[self.col_municipio].replace(
            mapeo_municipios, regex=True, inplace=True)
        self.df[self.col_dpto].replace(mapeo_dptos, regex=True, inplace=True)

        self.df.loc[self.df[self.col_dpto] == 'BOGOTA, D.C.',
                    self.col_municipio] = 'BOGOTA, D.C.'
        self.df.loc[self.df[self.col_municipio] ==
                    'BOGOTA, D.C.', self.col_dpto] = 'BOGOTA, D.C.'

        self.df.loc[(self.df[self.col_dpto] == 'BOLIVAR') &
                    (self.df[self.col_municipio].str.contains('CARTAG', na=False)), self.col_municipio] = 'CARTAGENA DE INDIAS'
        self.df.loc[(self.df[self.col_dpto] == 'SUCRE') &
                    (self.df[self.col_municipio].str.contains('VIEJO', na=False)), self.col_municipio] = 'SAN JOSE DE TOLUVIEJO'

    def get_municipios_malos(self, estandarizado=False):
        # no_presente = get_municipios_malos(df_icetex, 'municipio')
        '''USAR iloc'''
        sufijo = '_oficial' if estandarizado is True else ''
        no_presente = pd.merge(self.df, df_divipola,
                               left_on=[self.col_municipio, self.col_dpto],
                               right_on=['nombre_municipio' +
                                         sufijo, 'nombre_dpto'+sufijo],
                               how='left', validate='m:1',
                               indicator=True).query('_merge == "left_only"').drop(['_merge'], axis=1)
        return no_presente.index

    def verificar_divipola(self):
        """Retorna la diferencia de muni, y la de los dptos.
        """
        # self.procesar_columnas()
        no_presente = self.get_municipios_malos()
        # MUNICIPIOS
        errores_municipios = self.df.iloc[no_presente][[self.col_dpto, self.col_municipio]
                                                       ].dropna().value_counts().sort_index(level=1).to_frame('conteo')
        # DEPARTAMENTOS
        errores_departamentos = self.df.loc[~self.df[self.col_dpto].isin(df_divipola.nombre_dpto), [
            self.col_municipio, self.col_dpto]].value_counts().to_frame('conteo')
        print(
            f'Hay {errores_municipios.shape[0]} EERORES en municipios y {errores_departamentos.shape[0]} ERRORES en departamentos')
        return errores_municipios, errores_departamentos

    def corregir_muni_fuzzy(self, inplace=False, min_score=80):
        df = self.df.copy()

        no_presente = self.get_municipios_malos()
        df['municipio_estandar'] = df[self.col_municipio].mask(
            df.index.isin(no_presente))
        df['fuzzy'] = df.loc[df['municipio_estandar'].isna()].apply(
            lambda row: extraer_fuzzy(row.get(self.col_municipio), row.get(self.col_dpto), min_score=min_score), axis=1)

        if inplace is True:
            self.df[self.col_municipio] = df['municipio_estandar'].fillna(
                df['fuzzy'])
            print('Reemplazado')
        elif inplace == 'manual':
            # TODO poder hacer cambios manuales
            df['municipio_estandar'] = df['municipio_estandar'].fillna(
                df['fuzzy'])

        revisar = df.loc[df['municipio_estandar'].isna(), [self.col_dpto, self.col_municipio, 'fuzzy']]\
            .sort_values(self.col_municipio)
        revisar['conteo'] = revisar.groupby([self.col_dpto, self.col_municipio],
                                            dropna=False)['fuzzy'].transform('size')
        return revisar.drop_duplicates()

    def organizar_divipola(self, oficial=True, with_accents=False):
        '''EL Oficial le quita la modificaicon al codigo del departamento.'''
        df_codigo = procesar_municipio_depto(
            self.df, self.col_municipio, self.col_dpto, with_accents=with_accents)
        if oficial:
            df_codigo['codigo_dpto_modificado'] = df_codigo['codigo_dpto_modificado'].astype(
                'string').str[-2:]
            df_codigo.rename(
                columns={'codigo_dpto_modificado': 'codigo_dpto'}, inplace=True)
        return df_codigo

    def fill_capitales(self):
        # TODO
        df_capitales = get_df_capitales()
        df_capitales.departamento = df_capitales.departamento.apply(
            remove_accents)
        df_capitales.capital_departamento = df_capitales.capital_departamento.apply(
            remove_accents)
        df_capitales.rename(
            columns={'departamento': self.col_dpto}, inplace=True)
        self.df = pd.merge(self.df, df_capitales, on=self.col_dpto, how='left')
        self.df[self.col_municipio].replace('<NA>', pd.NA, inplace=True)
        self.df[self.col_municipio] = self.df[self.col_municipio].fillna(
            self.df.capital_departamento)
        self.df.loc[self.df[self.col_municipio] ==
                    'BOGOTA, D.C.', self.col_dpto] = 'BOGOTA, D.C.'


def get_Q_divipola(diccionario):

    q_divipola = Q()
    for dept, muni in diccionario.items():
        # Chequea si el diccionario efectivamente mapea a un municipio en la DB
        [ListItem.objects.get(nombre=municipio, list_type__codigo='muni', parent__nombre=dept.upper())
         for municipio in muni]
        q_divipola |= Q(departamento_residencia__nombre=dept.upper(),
                        municipio_residencia__nombre__in=muni)
    return q_divipola


def get_Q_estados(diccionario):
    q_estados = Q()
    for convocatoria, estados in diccionario.items():
        q_estados |= Q(convocatoria__nombre__iregex=convocatoria,
                       estado__codigo__in=estados)
    return q_estados


def make_query(database='default', return_df_aspirantes=False, respuestas=False, grupo_fila=['convocatoria_nombre'],
               grupo_columna=['estado__nombre'], totales=True, get_query = False,**kwgs):
    cols_query = ['programa', 'pk','convocatoria_nombre', 'puesto_llegada', 'nombres', 'apellidos', 'celular', 'telefono', 'genero_usuario', 'correo', 'tipo_documento_usuario',
                  'numero_documento', 'municipio_residencia_usuario', 'departamento_residencia_usuario', 'codigo_divipola', 'respuestas', 'estado_usuario', 'etiquetas',
                  'nivel_educacion', 'puesto_examen', 'aciertos', 'aciertos_parciales','situacion_vulnerabilidad_usuario',
                  'fecha_nacimiento', 'edad', 'estrato_usuario']
    query = Q()   # If you need OR operator use query |= Q() For AND operator use query &= Q()
    if 'divipola' in kwgs:
        query &= get_Q_divipola(kwgs.pop('divipola'))
    if 'convocatorias_estados' in kwgs:
        convocatorias_estados = kwgs.pop('convocatorias_estados')
        if not isinstance(convocatorias_estados, dict):
            raise TypeError('convocatorias_estados ',
                            'debe ser tipo diccionario.')
        query &= get_Q_estados(convocatorias_estados)
    if 'municipios' in kwgs:
        municipios = kwgs.pop('municipios')
        if not isinstance(municipios, list):
            raise TypeError('Municipios ', 'debe ser tipo lista.')
        query &= Q(municipio_residencia__nombre__in=municipios)
    if 'departamentos' in kwgs:
        departamentos = kwgs.pop('departamentos')
        if not isinstance(departamentos, list):
            raise TypeError('departamentos ', 'debe ser tipo lista.')
        query &= Q(departamento_residencia__nombre__in=departamentos)
    if 'genero' in kwgs:
        query &= Q(genero__codigo=kwgs.pop('genero'))
    if 'estados' in kwgs:
        estados = kwgs.pop('estados')
        if not isinstance(estados, list):
            raise TypeError('Estados ', 'debe ser tipo lista.')
        query &= Q(estado__codigo__in=estados)
    if 'tags' in kwgs:
        tags = kwgs.pop('tags')
        if not isinstance(tags, list):
            raise TypeError('Tags ', 'debe ser tipo lista.')
        query &= Q(tag__codigo__in=tags)
    if 'estratos' in kwgs:
        estratos = kwgs.pop('estratos')
        if not isinstance(estratos, list):
            raise TypeError('Los estratos ', 'debe ser tipo lista.')
        query &= Q(estrato__codigo__in=estratos)
    if 'programa' in kwgs:
        query &= Q(
            convocatoria__programa__nombre__iregex = kwgs.pop('programa'))
    if 'convocatoria' in kwgs:
        convocatoria = kwgs.pop("convocatoria")
        query &= Q(convocatoria__nombre__iregex=convocatoria)
    if 'custom_query' in kwgs:
        custom_query = kwgs.pop("custom_query")
        if isinstance(custom_query, dict):
            custom_query = Q(**custom_query)
        query &= custom_query
    if 'or_query' in kwgs:
        or_query = kwgs.pop("or_query")
        if isinstance(or_query, dict):
            or_query = Q(**or_query)
        query |= or_query
    exclude = False
    if 'exclude_query' in kwgs:
        exclude = True
        exclude_query = kwgs.pop("exclude_query")
        if isinstance(exclude_query, dict):
            exclude_query = Q(**exclude_query)
    mas_filtros = False
    if 'mas_filtros' in kwgs:
        mas_filtros = True
        mas_filtros_query = kwgs.pop("mas_filtros")
        if isinstance(mas_filtros_query, dict):
            mas_filtros_query = Q(**mas_filtros_query)
    if kwgs:
        raise TypeError('Unepxected kwargs provided: %s' % list(kwgs.keys()))

    resultado = ResultadoAspiranteConvocatoria.objects.using(database).\
        annotate(genero_usuario=F('genero__nombre'),
                 municipio_residencia_usuario=F(
            'municipio_residencia__nombre'),
        departamento_residencia_usuario=F(
            'departamento_residencia__nombre'),
        codigo_divipola=F(
            'municipio_residencia__pk'),
        convocatoria_nombre=F(
            'convocatoria__nombre'),
        estrato_usuario=F(
            'estrato__codigo'),
        tipo_documento_usuario=F(
            'tipo_documento__nombre'),
        nivel_educacion=F(
            'escolaridad__nombre'),
        pk=F(
            'pk'),
        estado_usuario=ArrayAgg(
            'estado__nombre', distinct=True),
        etiquetas=ArrayAgg(
            'tag__nombre', distinct=True),
        programa=F('convocatoria__programa__nombre'),
        situacion_vulnerabilidad_usuario = F('situacion_vulnerabilidad__nombre'),
    ).filter(query)

    if exclude is True:
        resultado = resultado.exclude(
            exclude_query
        )
    if mas_filtros is True:
        resultado = resultado.filter(
            mas_filtros_query
        )
    if get_query:
        import sqlparse
        sql_parsed = sqlparse.format(str(resultado.query), reindent=True, keyword_case='upper')
        print('---------\nQUERY SQL:\n', sql_parsed, '\n---------\n')
    print(resultado.count())
    if resultado.count() == 0:
        print('No hay resultados')
        return None, None
    groupby = [*grupo_fila, *
               grupo_columna] if grupo_columna is not None else [*grupo_fila]
    print(groupby)
    conteo = resultado.values(*groupby)\
        .annotate(conteo=Count('pk', distinct=True)).order_by('-conteo')

    df_conteo = pd.DataFrame.from_records(conteo)
    df_conteo.set_index(groupby, inplace=True)
    if grupo_columna is not None:
        df_conteo = df_conteo.unstack().sort_index()
    try:
        df_conteo = df_conteo.to_frame()
    except:
        pass
    try:
        df_conteo.columns = df_conteo.columns.droplevel()
    except:
        pass
    df_conteo.fillna(0, inplace=True)
    if totales is True:
        if df_conteo.shape[1] > 1:
            df_conteo['total'] = df_conteo.sum(axis=1)
        if df_conteo.shape[0] > 1:
            # Si el grupo_fila es mayor a dos debo crear una lista con la misma longitud del multiindice
            if len(grupo_fila) > 1:
                nombre_indices = tuple(['' for i in range(
                    len(df_conteo.index[0]) - 1)] + ['total'])  # ('','', 'total')
            else:
                nombre_indices = 'total'
            df_conteo.loc[nombre_indices, :] = df_conteo.sum(axis=0)
    df_conteo.index.name = ''  # TODO por poner un nombre bonito
    df_conteo = df_conteo.reset_index()
    df_conteo.columns = df_conteo.columns.astype(
        'string').fillna('No registra')
    df_conteo = df_conteo.convert_dtypes()
    df_conteo.iloc[:, 0] = df_conteo.iloc[:, 0].astype(
        'string').fillna('No registra')
    df_conteo.iloc[:, 0] = df_conteo.iloc[:, 0].apply(
        desestandarizar).str.upper()

    print('Conteo... DONE.')

    if return_df_aspirantes is True:
        print('extrayendo aspirantes..')
        df_aspirantes = pd.DataFrame.from_records(resultado.values())
        cols_interes = kwgs.get('cols_interes', df_aspirantes.columns)
        cols_interes = cols_query
        df_aspirantes = df_aspirantes[cols_interes]

        if respuestas is True:
            df_aspirantes.respuestas = df_aspirantes.respuestas.apply(
                lambda x: {} if pd.isnull(x) else x)
            df_respuestas = pd.json_normalize(df_aspirantes['respuestas'])
            df_aspirantes = pd.concat([df_aspirantes, df_respuestas], axis=1)
            df_aspirantes = df_aspirantes.filter(regex='^(?!.*(\.tipo))')
            df_aspirantes.columns = df_aspirantes.columns.str.replace(
                '\.opciones', '', regex=True)
            preguntas = Pregunta.objects.filter(
                ~Q(codigo__startswith='prueba'))

            df_aspirantes = df_aspirantes.apply(lambda s: s.apply(
                lambda s: s[0] if isinstance(s, list) and len(s) == 1 else s), axis=1)

            # Mapear las opciones de la pregunta de codigo a nombre
            preguntas_in_query = pd.json_normalize(
                df_aspirantes['respuestas'], max_level=0).columns
            for preg in Pregunta.objects.filter(codigo__in=preguntas_in_query, tipo__codigo='seleccion'):
                mapeo_opciones = {o.codigo: o.nombre for o in Opcion.objects.filter(
                    opcion_pregunta__codigo=preg.codigo)}
                df_aspirantes[preg.codigo] = df_aspirantes[preg.codigo].replace(
                    mapeo_opciones)
            # TODO QUE TAMBIEN REEMPLACE CUANDO ES UNA LISTA LA RESPUESTA TIPO MULTI-SELECT

            # Renombrar las columans segun el nombre de las respuestas
            mapeo_codigos_preguntas = {p.codigo: p.nombre for p in preguntas}
            df_aspirantes.rename(columns=mapeo_codigos_preguntas, inplace=True)

        df_aspirantes.insert(0, 'puesto_llegada',
                             df_aspirantes.pop("puesto_llegada"))
        df_aspirantes.rename(
            columns={'puesto_llegada': 'LLAVE MINTIC'}, inplace=True)

        if 'estado_usuario' in cols_interes:
            df_dummy = get_dummies_list(df_aspirantes.estado_usuario)
            df_dummy.replace({0: 'NO', 1: 'SI'}, inplace=True)
            df_aspirantes = pd.concat(
                [df_aspirantes, df_dummy], axis=1)

        if 'save' in kwgs:
            path = kwgs.get("numero_ruta")
            df_aspirantes.to_excel(path, index=False)

        return df_conteo, df_aspirantes
    # display(df_conteo)
    return df_conteo, None


def get_pickle(file, overwrite=False, **kwargs):
    if isinstance(file, str):
        file = Path(file)
    pickle = file.with_suffix('.pkl')
    if 'sheet_name' in kwargs:
        pickle = str(pickle.parent) + '\\' + \
            kwargs.get('sheet_name')+'_' + pickle.stem + pickle.suffix
        pickle = Path(pickle)
        print(pickle)
    if not pickle.exists() or overwrite:
        if file.suffix == '.csv':
            excel = pd.read_csv(file, **kwargs)
        elif file.suffix in ['.xlsx', '.xls']:
            excel = pd.read_excel(file, **kwargs)
        excel.to_pickle(pickle)
    return pd.read_pickle(pickle).convert_dtypes()


def get_preguntas_db(convocatoria=None):
    """
    Saca un dataframe con el codigo, nombre de la pregunta y un columna de las convocatorias (En lista). 
    Por defecto de todas las convocatorias sin embargo se puede especificar cual.
    """
    query = Q()
    # Sacar lista por pregunta.
    if convocatoria is not None:
        query &= Q(
            preguntas_convocatoria__nombre__contains=convocatoria)
    return pd.DataFrame(Pregunta.objects.filter(query)
                        .annotate(convocatoria=ArrayAgg('preguntas_convocatoria__nombre'))
                        .values('codigo', 'nombre', 'convocatoria')).sort_values('codigo').reset_index(drop=True)


class LoadDB:
    """[summary]
    load_df = LoadDB(df_aspirantes, 'Ciencia de datos - 2021-01')
    Secuencia:
        1.  load_df.get_datos_basicos()
        2.  load_df.get_columnas_listitem()
            Si se esta seguro que la diferencia debe agregarse:
                2.1.    load_df.agregar_listitems()
        3.  load_df.get_preguntas(). Para ver las preguntas presentes y las que deben de ser agregadas a la convocatoria.
            3.1. load_db.crear_preguntas_convocatoria() puede agregar las preguntas asi.
        4.  load_db.agregar_preguntas_convocatoria() --- Agrega las preguntas_presentes a la relacion con la convocatoria.
        5.  load_df.get_opciones_pregunta()
            Si se esta seguro que la diferencia debe agregarse:
                5.1.    load_db.agregar_opciones_pregunta()
        6. 
    Opcional:
        1. Para agregar respuestas use: load_df.agregar_respuestas()
    """
    campos_basicos = ['puesto_llegada', 'tipo_docu', 'numero_documento', 'nombres', 'apellidos', 'fecha_nacimiento', 'genero',
                      'edad', 'codigo_dpto_modificado', 'codigo_municipio', 'direccion', 'telefono', 'celular', 'correo',
                      'hora_envio_formulario', 'ocupacion', 'nivel_educacion', 'ocupacion', 'situacion_vulnerabilidad', 'estratos', 'puesto_examen', 'aciertos', 'aciertos_parciales']

    def __init__(self, df, convocatoria: str = None, examen=None, validar=True, database='default'):
        self.df = df.copy()
        self.db = database
        self.convocatoria = Convocatoria.objects.using(self.db).get(
            nombre__contains=convocatoria)
        self.examen = examen
        self.get_datos_basicos()
        self._procesar()
        self.preprocesar_df()
        if validar is True:
            self.df.numero_documento = pd.to_numeric(
                self.df.numero_documento).astype('Int64')
            if self.df.numero_documento.min() < 0:
                raise ValueError('No tiene sentido')
        if not (self.df.puesto_llegada.is_unique) or not (self.df.puesto_llegada.isna().sum() == 0):
            raise ValueError('Puesto de llegada no valido como llave.')

        # TODO Chequear si las fechas son formato fecha. Porque sino saca error y uno se demroa mucho buscandolo.
        # from pandas.api.types import is_datetime64_any_dtype as is_datetime
        # is_datetime(load_df.df.fecha_nacimiento)

    def preprocesar_df(self):
        self.df = procesar_df(self.df)
        self.df.columns = self.df.columns.str.replace('^desc_', '', regex=True)
        organizar_columnas(self.df)

    def mapear_columnas(self, mapeo_cols: dict):
        self.df.rename(columns=mapeo_cols, inplace=True)

    def eliminar_columnas(self, drop_cols):
        self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

    def _procesar(self):
        '''Determina las preguntas presentas y no presentes'''
        preguntas_existentes = get_preguntas_db().codigo.tolist()
        self.preguntas = check_difference(
            self.df.columns, self.datos_basicos_presentes)
        self.preguntas_no_presentes = check_difference(
            self.preguntas, preguntas_existentes)
        self.preguntas_presentes = self.preguntas[~self.preguntas.isin(
            self.preguntas_no_presentes)].to_list()

    def get_preguntas(self):
        self._procesar()
        print('Preguntas presentes:', '\n', self.preguntas_presentes)
        print('-'*15)
        print('Columnas NO registradas en la base de datos:\n',
              self.preguntas_no_presentes.tolist())

    def get_datos_basicos(self):
        dict_presente = {
            dato: dato in self.df.columns for dato in self.campos_basicos}
        df_presente = pd.DataFrame.from_dict(
            dict_presente, orient='index', columns=['Presente'])
        self.datos_basicos_presentes = df_presente.query(
            'Presente==True').index.to_series()
        df_presente = df_presente.join(self.df.isna().sum().rename('Faltantes')).convert_dtypes()
        self.df_datos_basicos = df_presente
        return df_presente.style.applymap(lambda x: 'color: green' if x else 'color: red', 
                                          subset=pd.IndexSlice[:, ['Presente']])

    def get_opciones_pregunta(self):
        '''Estandariza las opciones de seleccion. Y luego se revisa que informacion hay en la base de datos'''
        preg = Pregunta.objects.using(self.db).filter(
            codigo__in=self.preguntas_presentes, tipo__codigo='seleccion')
        codigos_preg = [p.codigo for p in preg]
        self._codigos_preg = codigos_preg
        print('Estandarizando las columnas: ', codigos_preg, end='\n\n')
        cols_con_lista = self.get_cols_lista()

        # Se chequea a ver si se puede eliminar las columnas
        checkeo = [col in codigos_preg for col in cols_con_lista]
        assert all(checkeo)
        only_in_codigos_preg = list(
            set(codigos_preg).difference(set(cols_con_lista)))
        print('Columnas multiselect: ', cols_con_lista)
        print('heyeyy', only_in_codigos_preg)
        for col_multi_select in cols_con_lista:

            self.df[col_multi_select] = self.df[col_multi_select].apply(lambda lista: [estandarizar(i) if isinstance(i, str)
                                                                                       else i for i in lista] if isinstance(lista, list) else lista)
        self.df[only_in_codigos_preg] = self.df[only_in_codigos_preg].apply(
            lambda s: s.apply(estandarizar), result_type="reduce")
        opciones_df = self.df.apply(
            lambda s: s.explode().unique(), result_type="reduce")[codigos_preg]
        if 'borrar' in codigos_preg:
            codigos_preg.remove('borrar')

        # if len(codigos_preg) == 1:
        #     opciones_df.drop(index='borrar', inplace=True)
        print(type(opciones_df))
        df_opciones_DB = pd.DataFrame(preg
                                      .annotate(opciones_DB=ArrayAgg('opciones__codigo')).values('codigo',
                                                                                                 'opciones_DB'))\
            .set_index('codigo')
        # return codigos_preg
        opciones_pregunta = opciones_df.to_frame(
            'opciones_df').join(df_opciones_DB)
        opciones_pregunta['opciones_df'] = opciones_pregunta['opciones_df'].apply(
            lambda lista: [i for i in lista if pd.notnull(i)] if isinstance(lista, (list, np.ndarray)) else [lista])
        opciones_pregunta['diferencia'] = opciones_pregunta['opciones_df'].map(
            set) - opciones_pregunta['opciones_DB'].map(set)
        opciones_pregunta.diferencia = opciones_pregunta.diferencia.where(
            opciones_pregunta.diferencia.apply(lambda x: len(x) != 0))
        return opciones_pregunta

    def get_cols_lista(self):
        # TODO MULTISELECT
        cols_con_lista = self.df.applymap(
            lambda x: isinstance(x, list)).any(axis=0)
        cols_con_lista = cols_con_lista.loc[cols_con_lista].index.tolist()
        return cols_con_lista

    @property
    def resumen(self):
        get_categories(self.df, limite=100).pipe(full_print)

    def crear_preguntas_convocatoria(self, nuevas_preguntas: dict, col_seleccion):
        """Crea nuevas preguntas y las relaciona con la convocatoria.

        Args:
            nuevas_preguntas (dict]): La llave es el codigo de la pregunta y el valor el nombre bonito.
            col_seleccion (list): lista de las preguntas que seran tipo seleccion. Si ninguna es de selección poner None
        """
        activo = ListItem.objects.using(self.db).get(
            codigo='activo', list_type__codigo='estados')

        for codigo, nombre in nuevas_preguntas.items():
            if col_seleccion is not None:
                tipo_preg = ListItem.objects.using(self.db).get(codigo='seleccion' if codigo in col_seleccion else 'abiertas',
                                                                list_type__codigo='tipo_preguntas')
            else:
                tipo_preg = ListItem.objects.using(self.db).get(
                    codigo='abiertas', list_type__codigo='tipo_preguntas')

            pregunta, created = Pregunta.objects.using(self.db).get_or_create(
                codigo=codigo, defaults=dict(estado=activo, nombre=nombre, tipo=tipo_preg))
            if created:
                print('Pregunta creada')
            self.convocatoria.preguntas.add(pregunta)

    def agregar_preguntas_convocatoria(self):
        """Agrega las preguntas_presentes en la tabla Preguntas, según el codigo a la relacion con la convocatoria.
        """
        print('-'*20)
        print('Agregando preguntas a la convocatoria...\n')
        self.get_preguntas()
        activo = ListItem.objects.using(self.db).get(
            codigo='activo', list_type__codigo='estados')
        print('\n')
        for codigo in self.preguntas_presentes:
            print(f'- {codigo} agregada.')
            pregunta, created = Pregunta.objects.using(self.db).get_or_create(
                codigo=codigo, defaults=dict(estado=activo))
            self.convocatoria.preguntas.add(pregunta)
        print('-'*20)

    def agregar_opciones_pregunta(self, agregar_todo=False, codigo: str = None):
        """Agrega y relaciona las opciones a las preguntas según el resultado en la columna Diferencia.

        Args:
            agregar_todo (bool, optional): Si desea que se mapee todo. Defaults to False.
            codigo (str, optional): Si solo desea insertar uno. Defaults to None.
        """
        if agregar_todo:
            for codigo_preg in self._codigos_preg:
                agregar_opciones(
                    self.df, codigo_pregunta=codigo_preg, db=self.db)
        elif codigo is not None:
            agregar_opciones(self.df, codigo_pregunta=codigo, db=self.db)
        self._procesar()

    def agregar_respuestas(self, llaves=None, sobreescribir=False):
        '''Agrega los campos json'''
        if llaves is None:
            llaves = self.df.puesto_llegada
        query_resultado = ResultadoAspiranteConvocatoria.objects.using(self.db).filter(convocatoria=self.convocatoria,
                                                                                       puesto_llegada__in=llaves)
        # assert self.get_opciones_pregunta()['diferencia'].isna().all()
        for index, row in self.df.iterrows():
            print(index, end=', ')
            row = row.where(row.notnull(), None)
            resultado = ResultadoAspiranteConvocatoria.objects.using(self.db).get(
                puesto_llegada=row.puesto_llegada, convocatoria=self.convocatoria)
            if resultado.respuestas and sobreescribir:
                raise ValueError('Ya hay respuestas!')
            resultado.respuestas = cargar_respuestas(
                row, self.preguntas_presentes)
            resultado.save()

    def cargar_db(self, estado_inicial=None, con_respuestas=True):
        for index, row in self.df.iterrows():
            print(index, end=', ')
            load_data(row, self.convocatoria, self.preguntas_presentes,
                      self.examen, estado=row.get('estado_individual', estado_inicial), con_respuestas=con_respuestas, database=self.db)

    def _get_listitem_df(self):
        df = (pd.DataFrame(ListType.objects.using(self.db).annotate(num_items=Count('listitem__codigo'),
                                                                    items_DB=ArrayAgg('listitem__codigo'))
                           .values('codigo', 'nombre', 'items_DB', 'num_items')
                           ))
        return df

    def get_columnas_listitem(self) -> pd.DataFrame:
        df_listitem = self._get_listitem_df().set_index('codigo')['items_DB']
        for col in df_listitem.index[df_listitem.index.isin(self.df.columns)]:
            print('Estandarizado', col)
            self.df[col] = self.df[col].astype('string').apply(estandarizar)

        df_unicos = self.df.loc[:, self.df.columns.isin(
            df_listitem.index)].reset_index(drop=False)
        df_unicos = df_unicos.apply(lambda s: s.explode().unique(), result_type="reduce").to_frame(
            'items_df')  # .set_index('index')
        df_unicos.drop(index='index', inplace=True)

        columnas_listitem = df_unicos.join(df_listitem)
        # Quito los valores nulos de la lista
        columnas_listitem['items_df'] = columnas_listitem['items_df'].apply(
            lambda lista: [i for i in lista if pd.notnull(i)])
        columnas_listitem['items_df'] = columnas_listitem['items_df'].apply(
            np.sort)
        columnas_listitem['items_DB'] = columnas_listitem['items_DB'].apply(
            np.sort)
        columnas_listitem['diferencia'] = columnas_listitem['items_df'].map(
            set) - columnas_listitem['items_DB'].map(set)
        columnas_listitem.diferencia = columnas_listitem.diferencia.where(
            columnas_listitem.diferencia.apply(lambda x: len(x) != 0))
        self.columnas_listitem = columnas_listitem
        return columnas_listitem

    def agregar_listitems(self):
        for codigo, data in self.columnas_listitem.diferencia.dropna().iteritems():
            print('Agregado', list(data))
            agregar_db_list_items(list(data), codigo, database=self.db)

    def update(self, llave='puesto_llegada', fields_to_change=None):
        """
        Actualiza los campos especificados en la base de datos.
        """
        if not isinstance(fields_to_change, list):
            raise TypeError(
                'Especifique una lista de campos que quiere cambiar.')
        for index, row in self.df.iterrows():
            row = row.where(row.notnull(), None)
            print(index, end=', ')
            obj = ResultadoAspiranteConvocatoria.objects.using(self.db).get(
                puesto_llegada=row[llave], convocatoria=self.convocatoria)

            for key in fields_to_change:
                setattr(obj, key, row[key])
            obj.save()

    def agregar_estados(self):
        pass
        # TODO hacer un wrapper de la funcion agregar_estados ya definida
        agregar_estados()


def load_data(row, convocatoria, preguntas_convocatoria, examen, estado='aspirante', 
              con_respuestas=True, database=None):
    # -----
    row = row.where(row.notnull(), None)
    datos_personales = dict(
        departamento_residencia=ListItem.objects.using(database).get(pk=row.get(
            'codigo_dpto_modificado'), list_type__codigo='depart') if row.get('codigo_dpto_modificado') else None,
        municipio_residencia=ListItem.objects.using(database).get(pk=row.get(
            'codigo_municipio'), list_type__codigo='muni') if row.get('codigo_municipio') else None,
        genero=ListItem.objects.using(database).get(codigo=row.get(
            'genero'), list_type__codigo='genero') if row.get('genero') else None,
        escolaridad=ListItem.objects.using(database).get(codigo=row.get(
            'nivel_educacion'), list_type__codigo='nivel_educacion') if row.get('nivel_educacion') else None,
        estrato=ListItem.objects.using(database).get(codigo=row.get(
            'estratos'), list_type__codigo='estratos') if row.get('estratos') else None,
        nombres=row.get('nombres'),
        apellidos=row.get('apellidos'),
        telefono=int(row.get('telefono')) if isinstance(
            row.get('telefono'), float) else row.get('telefono'),
        celular=int(row.get('celular')) if isinstance(
            row.get('celular'), float) else row.get('celular'),
        situacion_vulnerabilidad=ListItem.objects.using(database).get(codigo=row.get(
            'situacion_vulnerabilidad')) if row.get('situacion_vulnerabilidad') else None,
        correo=row.get('correo'),
        edad=row.get('edad'),
        fecha_nacimiento=row.get('fecha_nacimiento'),
        direccion=row.get('direccion'),
        # ocupacion=row.get('ocupacion') if row.get('ocupacion') else None,
    )

    datos_unicos = dict(numero_documento=int(row.numero_documento) if isinstance(row.numero_documento, float) else row.numero_documento,
                        tipo_documento=ListItem.objects.using(database).get(codigo=row.tipo_docu,
                        list_type__codigo='tipo_docu'))

    # ---------
    # TODO cambiarlo para q sea con  update_or_create
    ciudadano, _ = Ciudadano.objects.using(database).get_or_create(**datos_unicos,
                                                                   defaults=datos_personales)

    a, _ = Aspirante.objects.using(database).get_or_create(ciudadano=ciudadano,
                                                           tipo=ListItem.objects.using(database).get(codigo='general'))

    # TODO mirar los timezones
    tz = 'America/Bogota'
    # tz = a.created.tzinfo
    # row[preguntas_convocatoria] = row[preguntas_convocatoria].apply(lambda s:str(s) if isinstance(s,datetime.date) else s)
    resultado, _ = ResultadoAspiranteConvocatoria.objects.using(database).get_or_create(aspirante=a,
                                                                                        convocatoria=convocatoria,
                                                                                        puesto_llegada=row['puesto_llegada'],
                                                                                        defaults=dict(
                                                                                            puesto_examen=int(row.get('puesto_examen')) if row.get(
                                                                                                'puesto_examen') else None,
                                                                                            ocupacion=row.get('ocupacion') if row.get(
                                                                                                'ocupacion') else None,
                                                                                            examen=examen,
                                                                                            hora_recibido=row.get('hora_envio_formulario').tz_localize(
                                                                                                tz) if row.get('hora_envio_formulario') else None,
                                                                                            aciertos_parciales=row.get(
                                                                                                'aciertos_parciales'),
                                                                                            aciertos=row.get(
                                                                                                'aciertos'),
                                                                                            respuestas=cargar_respuestas(
                                                                                                row, preguntas_convocatoria) if con_respuestas is True else None,
                                                                                            **datos_unicos,
                                                                                            validacion_documento=True if REGEX_ID.match(
                                                                                                str(datos_unicos['numero_documento'])) else False,
                                                                                            validacion_correo=validar_correo(
                                                                                                datos_personales['correo']),
                                                                                            validacion_celular=True if REGEX_CELULAR.match(
                                                                                                str(datos_personales['celular'])) else False,
                                                                                            **datos_personales)
                                                                                        )

    e, created = Estado.objects.using(database).get_or_create(estado=ListItem.objects.using(database).get(codigo=estado, list_type__codigo='estados_usuarios'),
                                                              proceso_aspirante=resultado,
                                                              defaults=dict(
                                                                  origen_estado=f'Inscripción a la {convocatoria.nombre}')
                                                              )


def plot_na(df):
    df_na = df.copy()
    df_na.columns = df_na.columns.astype('string')
    df_na.reset_index(drop=True, inplace=True)
    if not df.isna().values.any():
        raise TypeError('No hay datos nulos en el dataframe.')

    displot = sns.displot(
        data=df.isna().melt(value_name='Faltante').replace(
            {True: 'SI', False: 'NO'}),
        y='variable',
        hue='Faltante',
        multiple='fill',
        palette=['k', 'red'],
        # aspect = 1.2,
        height=15,
        legend=False,
        #         alpha = 0.7
    )
    displot.set(xlabel='Proporción', ylabel='')
    plt.legend(labels=['SI', 'NO'], loc='upper center', ncol=2)
    plt.tight_layout()
    df_na_groups = df_na.groupby(
        df_na.index // 100).apply(lambda x: x.isna().sum())
    fig, ax = plt.subplots(figsize=(15, 16))
    sns.heatmap(df_na_groups.T, cbar=True,
                cbar_kws=dict(shrink=0.2, aspect=20),
                ax=ax,
                mask=df_na_groups.T == 0, cmap='viridis_r',
                yticklabels=True, xticklabels=False,
                )
    ax = plt.gca()
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top',)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(),
                       rotation=90, ha='center', fontsize=10)
    ax.hlines(range(1, int(ax.get_ylim()[0])),
              *ax.get_xlim(), color='gray', lw=0.8)
    plt.tight_layout()
    # plt.tick_params(roration=90)


def check_query(objs, get_items=False):
    "https://github.com/django/django/blob/1024b5e74a7166313ad4e4975a15e90dccd3ec5f/django/contrib/admin/utils.py#L105"
    "https://stackoverflow.com/questions/26807858/how-can-i-check-what-objects-will-be-cascade-deleted-in-django"
    "https://stackoverflow.com/questions/12158714/how-to-show-related-items-using-deleteview-in-django"
    from django.contrib.admin.utils import NestedObjects
    from django.db import router

    if not hasattr(objs, '__iter__'):
        objs = [objs]
    # using = router.db_for_write(objs[0]._meta.model) #Seria el ideal si tuviera un router bien definido
    using = objs[0]._state.db
    print(using)
    collector = NestedObjects(using=using)
    collector.collect(objs)
    model_count = {model._meta.verbose_name_plural: len(
        objs) for model, objs in collector.model_objs.items()}
    if get_items:
        to_delete = collector.nested()  # items que seran borrados
        return model_count, to_delete
    return model_count


def eliminar_codigosRepetidosPreguntas(codigo_pre):
    """Eliminaba los codigos de pregunta para poder poner el constraint.
    """
    opciones = Opcion.objects.filter(codigo=codigo_pre)
    if opciones.count() > 1:
        for pregunta in opciones[0].opcion_pregunta.all():
            pregunta.opciones.remove(
                Opcion.objects.filter(codigo=codigo_pre)[0])
            pregunta.opciones.add(Opcion.objects.filter(codigo=codigo_pre)[1])
    Opcion.objects.filter(codigo=codigo_pre).annotate(
        num_preguntas=Count('opcion_pregunta')).filter(num_preguntas=0).delete()
    Opcion.objects.get(codigo=codigo_pre).opcion_pregunta.all()


def organizar_respuestas(a, drop_na=False):
    # a es la columna q tiene las puestas
    df_aspirantes = pd.json_normalize(a)
    df_aspirantes = df_aspirantes.filter(regex='^(?!.*(\.tipo))')
    df_aspirantes.columns = df_aspirantes.columns.str.replace(
        '\.opciones', '', regex=True)
    preguntas = Pregunta.objects.filter(
        ~Q(codigo__startswith='prueba'))

    df_aspirantes = df_aspirantes.apply(lambda s: s.apply(
        lambda s: s[0] if isinstance(s, list) and len(s) == 1 else s), axis=1)

    # Mapear las opciones de la pregunta de codigo a nombre
    preguntas_in_query = pd.json_normalize(
        a, max_level=0).columns
    for preg in Pregunta.objects.filter(codigo__in=preguntas_in_query, tipo__codigo='seleccion'):
        mapeo_opciones = {o.codigo: o.nombre for o in Opcion.objects.filter(
            opcion_pregunta__codigo=preg.codigo)}
        df_aspirantes[preg.codigo] = df_aspirantes[preg.codigo].replace(
            mapeo_opciones)
    # TODO QUE TAMBIEN REEMPLACE CUANDO ES UNA LISTA LA RESPUESTA TIPO MULTI-SELECT

    # Renombrar las columans segun el nombre de las respuestas
    mapeo_codigos_preguntas = {p.codigo: p.nombre for p in preguntas}
    df_aspirantes.rename(columns=mapeo_codigos_preguntas, inplace=True)
    df_respuestas = df_aspirantes.T
    if drop_na is True:
        df_respuestas.dropna(inplace=True)
    df_respuestas.sort_index(inplace=True)
    return df_respuestas.to_dict()[0]


def verificar_integridad(df, col):
    '''assert verificar_integridad(load_db.df, 'puesto_llegada')'''
    resultado = df[col].is_unique and df[col].notnull().all()
    if resultado:
        print(f'La columna {col} no tiene nulos y es unica.')
    return resultado


def verificar_integridad(df, col):
    '''Verifica ssi una columna es unica y no tiene nulos

    assert verificar_integridad(load_db.df, 'puesto_llegada')
    '''
    resultado = df[col].is_unique and df[col].notnull().all()
    if resultado:
        print(f'La columna {col} no tiene nulos y es unica.')
    return resultado


def organizar_activos(db = 'servidor'):
    Tag.objects.using(db).filter(resultado_aspirante_tag__convocatoria__nombre__iregex = 'ruta \d - 2021',
                            etiqueta__nombre = 'Activo', 
                            ).delete()
    
    query_activos = ResultadoAspiranteConvocatoria.objects.using(db).filter(estado__codigo = 'beneficiario', 
                                    convocatoria__nombre__iregex = 'ruta \d - 2021')\
                                        .exclude(estado__codigo = 'desertor')
    print("Número de activos: ", query_activos.count())
    agregar_etiquetas('Activo', query_resultado=query_activos, descripcion='Script para agregar etiqueta de Activo')