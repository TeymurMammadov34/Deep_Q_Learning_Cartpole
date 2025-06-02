# 🧠 Deep Q-Learning ile CartPole Ajan Eğitimi
Bu proje, OpenAI Gym’in klasik kontrol problemlerinden biri olan CartPole-v1 ortamında, Deep Q-Learning (DQL) algoritmasını kullanarak bir ajan eğitmeyi amaçlamaktadır. Hedef, bir çubuğu dik tutmaya çalışan bir ajanın çevreyle etkileşim kurarak optimum politikayı öğrenmesini sağlamaktır.

## 🚀 Projenin Amacı
CartPole ortamında:

- Ajan, bir arabanın üstündeki çubuğun düşmemesi için sağa veya sola hareket eder.

- Çubuğun mümkün olan en uzun süre dengede tutulması hedeflenir.

- Ajan, durum bilgilerine bakarak hangi eylemi seçeceğini öğrenir.


## 🛠 Kullanılan Teknolojiler
- Python 3.x

- OpenAI Gym – RL ortamı

- TensorFlow / Keras – Derin sinir ağı modeli

- NumPy – Sayısal işlemler

- TQDM – Eğitim ilerlemesi için ilerleme çubuğu

- collections.deque ve random – Deneyim tekrar belleği


## 🧠 Deep Q-Learning Bileşenleri
- Experience Replay (Deneyim Tekrarı):
Ajanın deneyimlerini hafızada tutarak rastgele örneklerle öğrenmesi, korelasyonları azaltır ve öğrenmeyi daha kararlı hale getirir.

- ε-Greedy Politikası:
Ajan başlangıçta rastgele hareket ederek keşif yapar (exploration). Zamanla öğrendiği bilgiye göre daha bilinçli hareket eder (exploitation).

- Q-Network (Derin Sinir Ağı):
Girdi olarak ortamın durum vektörünü alır, çıktı olarak her eylem için Q-değerini (beklenen ödül) tahmin eder.


## 🧩 Model Mimarisi
Giriş Katmanı : 4 nöron  → (CartPole gözlem uzayı)

Gizli Katman 1: 48 nöron → ReLU aktivasyon

Gizli Katman 2: 24 nöron → ReLU aktivasyon

Çıkış Katmanı : 2 nöron  → Lineer aktivasyon (eylem sayısı: sağa veya sola hareket)


## ⚙️ Kurulum
Aşağıdaki komutlarla gerekli kütüphaneleri yükleyebilirsiniz:
```bash
pip install numpy tensorflow gym tqdm
```

## 🏋️‍♂️ Eğitimi Başlatma
```python
episodes = 5  # Eğitim süresini artırmak isterseniz bu değeri yükseltebilirsiniz

for e in range(episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1, 4])
    done = False
    time_t = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay()  # Ajan öğrenmesini burada gerçekleştirir
        time_t += 1

    print(f"Episode {e+1}/{episodes} finished after {time_t} timesteps")
```

## 🧪 Eğitilmiş Ajanı Test Etme
Aşağıdaki kod parçası ile eğitilmiş ajanı görsel olarak test edebilirsiniz:
```python
trained_model = agent

env = gym.make("CartPole-v1", render_mode="human")  # Ortam insan gözüyle izlenebilir şekilde başlatılır
state = env.reset()[0]
state = np.reshape(state, [1, 4])

time_t = 0

while True:
    env.render()  # Ortamı ekranda gösterir
    action = trained_model.act(state)  # Ajanın kararı
    next_state, reward, done, _, _ = env.step(action)  # Ortamla etkileşim
    next_state = np.reshape(next_state, [1, 4])
    state = next_state
    time_t += 1
    print(f"Time: {time_t}")
    time.sleep(0.5)  # Gözle izlemek için yavaşlatma

    if done:
        break

print("Done")

```

## 📈 Geliştirme Önerileri
- Double DQN veya Dueling DQN gibi gelişmiş algoritmalarla deneyin.
- Ortama reward shaping ekleyerek daha iyi performans elde edebilirsiniz.
- Eğitim verilerini grafiksel olarak izlemek için Matplotlib ile ödül/episode çizimi yapabilirsiniz.

## 🧾 Lisans
Bu proje **MIT** lisansı ile lisanslanmıştır. İstediğiniz gibi kullanabilir, geliştirebilir ve paylaşabilirsiniz.
