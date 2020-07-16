import React, { Component } from 'react';
import Form from 'react-bootstrap/Form';
import RangeSlider from 'react-bootstrap-range-slider';
import Button from 'react-bootstrap/Button';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';

import './ConfigPanel.css';

const WithTooltip = (props) => (
    <OverlayTrigger
        placement={props.placement}
        delay={{ show: 500 }}
        overlay={
            <Tooltip>
                {props.text}                                                  
            </Tooltip>
        }
    >
        {props.children}
    </OverlayTrigger>
)

WithTooltip.defaultProps = {
    placement: "top"
}

export default class ConfigPanel extends Component {
    constructor(props) {
        super(props);
        this.state = {
            recordTypes: {
                tifu: false,
                wp: false,
                phc: true
            },
            maxGuessingTimeEnabled: false,
            maxGuessingTime: 30
        };
    }

    recordTypeToggleChange = (event) => {
        this.setState({
            recordTypes: {
                ...this.state.recordTypes,
                [event.target.id]: event.target.checked
            }
        });
    }

    maxGuessingTimeChange = (event) => {
        this.setState({
            maxGuessingTime: event.target.value
        });
    }

    maxGuessingTimeToggleChange = (event) => {
        this.setState({
            maxGuessingTimeEnabled: event.target.checked
        });
    }

    render() {
        return (    
            <div className="settings-panel">
                <h4 className="text-center mt-2 font-weight-bold">settings</h4>
                <Form className="mt-4">
                    <Form.Group>
                        <Form.Label className="font-weight-bold">types of posts?</Form.Label>
                        <div>
                            <Form.Check id="tifu" inline custom className="record-type-checkbox">
                                <Form.Check.Input type="checkbox"
                                    onChange={this.recordTypeToggleChange}
                                    checked={this.state.recordTypes.tifu} />
                                <WithTooltip text="today i fucked up">
                                    <Form.Check.Label>tifu</Form.Check.Label>
                                </WithTooltip>
                            </Form.Check>
                            <Form.Check id="wp" inline custom className="record-type-checkbox">
                                <Form.Check.Input type="checkbox"
                                    onChange={this.recordTypeToggleChange}
                                    checked={this.state.recordTypes.wp} />
                                <WithTooltip text="writing prompts">
                                    <Form.Check.Label>wp</Form.Check.Label>
                                </WithTooltip>
                            </Form.Check>
                            <Form.Check id="phc" inline custom className="record-type-checkbox">
                                <Form.Check.Input type="checkbox"
                                    onChange={this.recordTypeToggleChange}
                                    checked={this.state.recordTypes.phc} />
                                <WithTooltip text="pornhub comments">
                                    <Form.Check.Label>phc</Form.Check.Label>
                                </WithTooltip>
                            </Form.Check>  
                        </div>
                    </Form.Group>
                    <Form.Group className="mt-4">
                        <Form.Check id="max-guessing-time-toggle" custom className="record-type-checkbox">
                            <Form.Check.Input type="checkbox" 
                                onChange={this.maxGuessingTimeToggleChange}
                                checked={this.state.maxGuessingTimeEnabled} />
                            <Form.Check.Label className="font-weight-bold"
                                disabled={!this.state.maxGuessingTimeEnabled}
                            >
                                max guessing time (seconds)
                            </Form.Check.Label>
                        </Form.Check>
                        <RangeSlider disabled={!this.state.maxGuessingTimeEnabled} value={this.state.maxGuessingTime}
                            min="0" max="60" className="max-guessing-time-range-slider"
                            onChange={this.maxGuessingTimeChange}
                        />
                    </Form.Group>
                    <div className="text-center mt-4">
                        <Button size="lg" className="ready-btn" onClick={() => this.props.action(this.state)}>
                            ready!
                        </Button>    
                    </div> 
                </Form>
            </div>         
        )
    }
}