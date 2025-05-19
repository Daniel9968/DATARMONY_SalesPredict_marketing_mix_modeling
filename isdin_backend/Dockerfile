# Usa una imagen base de Python 3.11
FROM python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Copia el resto del código de la aplicación
COPY . .

# Instala las dependencias
RUN pip install -r requirements.txt

# Comando para ejecutar el script
CMD ["python", "main.py"]