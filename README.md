# covid-django

Ejemplo mínimo de cómo levantar un servidor web con Django.
Para armarlo, me basé en 
[https://developer.mozilla.org/en-US/docs/Learn/Server-side]

Para construir este sitio
Para crear el proyecto cree la carpeta raíz y ejecuté las siguientes 
instrucciones:

```
> django-admin startproject webcovidserver
> cd webcovidserver
> python3 manage.py startapp webcovid
```


Luego, 

* modifiqué los archivos `webcovidserver/settings.py` 
para agregar la *aplicación* `webcovid`.
* modifiqué `webcovidserver/settings.py` para agregar las 
  URLs
* armé 
  - `webcovid/models.py` que define el acceso a los datos
  -  `webcovid/views.py` para definir como mostrar esos datos
  - `webcovid/templates/webcovid/index.html` -> la template de la página

Finalmente,  entrando en la carpeta `/webcovidserver`

* `python3 manage.py makemigrations` -> prepara la configuración de la base de datos
   del sitio
* `python3 manage.py migrate`  -> genera la base de datos
* `python3 manage.py runserver` -> inicia el server.



