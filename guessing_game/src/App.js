import React, { Component } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';

import HomePage from './pages/HomePage';
import GamePage from './pages/game/GamePage';
import './App.css';

export default class App extends Component {
    render() {
        return (
            <BrowserRouter>
                <Switch>
                    <Route path="/" component={HomePage} exact />
                    <Route path="/play" component={GamePage} exact />
                </Switch>
            </BrowserRouter>
        )
    }
}