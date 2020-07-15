import React, { Component } from 'react'
import ConfigPage from './ConfigPage';

export default class GamePage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            isConfigPageVisible: true
        };
    }

    onConfigPageReady = (configState) => {
        this.setState({
            gameConfig: configState,
            isConfigPageVisible: false
        });
    }

    render() {
        if (this.state.isConfigPageVisible) {
            return ( <ConfigPage action={this.onConfigPageReady} /> );
        }
        
        return (
            <div>

            </div>
        );
    }
}
