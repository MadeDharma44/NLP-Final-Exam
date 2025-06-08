// Inisialisasi peta
const map = L.map('map').setView([-2.5489, 118.0149], 5); // Koordinat Indonesia

// Tambahkan tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap'
}).addTo(map);
