import multiprocessing
import random
import matplotlib.pyplot as plt
import numpy as np

# Modulos neceserios para la implementación de multiprocesos, la generación aleatoria de puntos y la grafa visual del problema


'''Función que va a ejecutar cada proceso y en la que se cogen una fraccion de los puntos originales (se ve mas adelante).
En esta función se generan aleatoriamente puntos dentro de un rango (0,1) con la función random.random() para simular un cuadrado y luego
se verifica que estos puntos esten o no dentro de un circlo unitario. En el caso de starlo se incrementa el contador por uno. Al realizar 
la simulación Monte Carlo para calcular la parte de π y almacena el resultado en shared_dict."""
'''
def monte_carlo_pi_worker(n_puntos, shared_dict, lock, process_id):
    # Contador inicualizado en 0
    contador = 0
    # generación de puntos en un rango de n_puntos ( el número de puntos asignados para cada proceso)
    for _ in range(n_puntos):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1: # verificación de pertenencia al círculo y aumento del contador en este caso.
            contador += 1
    # Uso de lock para sincronizar el acceso al diccionario compartido.
    with lock:
        shared_dict[process_id] = contador # Diccionario clave valor de cada proceso (id_process) y su contador

def main():
    total_puntos = 10_000_000 # Número total de puntos para la simulación
    n_procesos = multiprocessing.cpu_count()  # Número de procesos basado en núcleos disponibles
    puntos_por_proceso = total_puntos // n_procesos #Número de puntos por proceso

    # Crear un Manager para gestionar objetos compartidos entre procesos
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()  # Diccionario compartido para almacenar resultados (gestionado por el manager)
    lock = manager.Lock()          # Lock para sincronización entre procesos (Evita condiciones de carrera)

    procesos = []# Lista de los procesos inicialmente vacía
    # Crear y arrancar los procesos cada uno con su id y su coleccion de puntos para la función monte_carlo_pi_worker
    for i in range(n_procesos):
        p = multiprocessing.Process(target=monte_carlo_pi_worker,
                                    args=(puntos_por_proceso, shared_dict, lock, i))
        procesos.append(p)
        p.start()

    # Esperar a que todos los procesos terminen
    for p in procesos:
        p.join()

    # Reducir los resultados: sumar todos los contadores almacenados en shared_dict 
    total_contador = sum(shared_dict.values())
    pi_estimado = 4 * total_contador / total_puntos # Realiza la simulación Monte Carlo para calcular la parte de π 
    print(f"Valor estimado de π: {pi_estimado}")

'''
Función para la representación visual de la simulación Monte Carlo.
Genera un número reducido de puntos y los clasifica en:
   - Dentro del cuarto de círculo (azul)
   - Fuera del cuarto de círculo (rojo)

Además, se dibuja la curva del cuarto de círculo para referencia.'''

def plot_simulacion(n_puntos=1000):
    puntos_dentro = []
    puntos_fuera = []
    for _ in range(n_puntos):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            puntos_dentro.append((x, y))
        else:
            puntos_fuera.append((x, y))
    fig, ax = plt.subplots()
    if puntos_dentro:
        xs, ys = zip(*puntos_dentro)
        ax.scatter(xs, ys, c='blue', s=10, label='Dentro del cuarto de círculo')
    if puntos_fuera:
        xs, ys = zip(*puntos_fuera)
        ax.scatter(xs, ys, c='red', s=10, label='Fuera del cuarto de círculo')
    # Dibujar la curva del cuarto de círculo usando valores de theta entre 0 y pi/2.
    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='green', linewidth=2, label='Curva del cuarto de círculo')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Simulación Monte Carlo para estimar π")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
    plot_simulacion()
