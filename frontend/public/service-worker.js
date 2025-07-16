const cacheName = 'swasthya-sathi-v1';
const staticAssets = [
  '/',
  '/index.html',
  '/static/css/main.8864bffe.css',
  '/static/js/main.b63a4d26.js',
  '/favicon.ico',
  '/manifest.json',
  '/logo192.png',
  '/logo512.png'
];

// Installation event: Cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(cacheName).then((cache) => {
      console.log('Caching static assets');
      return cache.addAll(staticAssets);
    })
  );
});

// Activation event: Clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== cacheName) {
            console.log('Clearing old cache:', name);
            return caches.delete(name);
          }
        })
      );
    })
  );
});

// Fetch event: Serve cached content or fetch from network
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        if (
          event.request.url.startsWith('http://localhost:5000') &&
          networkResponse.status === 200
        ) {
          return caches.open(cacheName).then((cache) => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
        }
        return networkResponse;
      }).catch(() => {
        if (
          !event.request.url.endsWith('.css') &&
          !event.request.url.endsWith('.js')
        ) {
          return caches.match('/offline.html');
        }
        return null;
      });
    })
  );
});

self.addEventListener('sync', (event) => {
  if (event.tag === 'log-water-intake') {
    event.waitUntil(
      fetch('http://localhost:5000/api/water/log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          amount: 0 // Replace with actual value from IndexedDB
        })
      })
        .then((response) => {
          if (response.ok) {
            console.log('Water intake logged successfully via background sync.');
          } else {
            console.error(
              'Failed to log water intake via background sync:',
              response.status
            );
          }
        })
        .catch((error) => {
          console.error(
            'Error sending water intake log via background sync:',
            error
          );
        })
    );
  } else if (event.tag === 'complete-habit') {
    event.waitUntil(
      caches.open(cacheName).then(() => {
        const dbName = 'swasthyaSathiDB';
        const storeName = 'pendingHabitCompletions';
        return new Promise((resolve, reject) => {
          const request = indexedDB.open(dbName, 1);
          request.onerror = (event) => {
            console.error('IndexedDB error:', event.target.errorCode);
            reject(null);
          };
          request.onsuccess = (event) => {
            const db = event.target.result;
            const transaction = db.transaction(storeName, 'readwrite');
            const store = transaction.objectStore(storeName);
            const getRequest = store.get(1);
            getRequest.onsuccess = () => {
              const pendingHabitCompletion = getRequest.result;
              if (pendingHabitCompletion) {
                const { habitId, completionDate } = pendingHabitCompletion;
                fetch('http://localhost:5000/api/habits/complete', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json'
                  },
                  body: JSON.stringify({
                    habit_id: habitId,
                    completionDate: completionDate
                  })
                })
                  .then((response) => {
                    if (response.ok) {
                      console.log('Habit completed successfully via background sync.');
                      store.delete(1);
                    } else {
                      console.error(
                        'Failed to complete habit via background sync:',
                        response.status
                      );
                    }
                    resolve();
                  })
                  .catch((error) => {
                    console.error(
                      'Error sending habit completion via background sync:',
                      error
                    );
                    resolve();
                  });
              } else {
                console.log('No pending habit completion to sync (from IndexedDB).');
                resolve();
              }
            };
          };
          request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(storeName)) {
              db.createObjectStore(storeName);
            }
          };
        });
      })
    );
  }
});

self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'get-daily-reminders') {
    event.waitUntil(
      fetch('http://localhost:5000/api/reminders/daily')
        .then((response) => response.json())
        .then((reminders) => {
          console.log('Fetched daily reminders in the background:', reminders);
        })
        .catch((error) => {
          console.error('Failed to fetch daily reminders during periodic sync:', error);
        })
    );
  }
});

self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'SwasthyaSathi Update';
  const options = {
    body: data.body || 'Check out the latest information!',
    icon: '/logo192.png'
  };
  event.waitUntil(self.registration.showNotification(title, options));
});