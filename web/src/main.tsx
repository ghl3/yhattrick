import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./styles.css";
import App from "./App";
import GamesIndex from "./routes/GamesIndex";
import GameView from "./routes/GameView";
import Players from "./routes/Players";
import Player from "./routes/Player";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <GamesIndex /> },
      { path: "game/:gameId", element: <GameView /> },
      { path: "players", element: <Players /> },
      { path: "player/:id", element: <Player /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
