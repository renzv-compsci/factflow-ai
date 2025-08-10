import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
      <div className="bg-white p-8 rounded shadow-lg text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">
          Tailwind CSS is <span className="text-green-500">Working!</span>
        </h1>
        <p className="text-gray-600">If you see styles and colors, Tailwind is set up correctly 🎉</p>
      </div>
    </div>
  );
}

export default App;
