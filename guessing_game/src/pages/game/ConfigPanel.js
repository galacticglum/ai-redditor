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
            maxGuessingTime: 30,
            isValid: true
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

    onReady = () => {
        // Reset the validation state of the form.
        this.setState({isValid: true});
        // Make sure that at least one record type is selected
        const noRecordTypesSelected = Object.keys(this.state.recordTypes).every(k => !this.state.recordTypes[k]);
        if (noRecordTypesSelected) {
            this.setState({isValid: false});
            return;
        }

        this.props.action(this.state);
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
                                    checked={this.state.recordTypes.tifu} isInvalid={!this.state.isValid} />
                                <WithTooltip text="today i fucked up">
                                    <Form.Check.Label>tifu</Form.Check.Label>
                                </WithTooltip>
                            </Form.Check>
                            <Form.Check id="wp" inline custom className="record-type-checkbox">
                                <Form.Check.Input type="checkbox"
                                    onChange={this.recordTypeToggleChange}
                                    checked={this.state.recordTypes.wp} isInvalid={!this.state.isValid} />
                                <WithTooltip text="writing prompts">
                                    <Form.Check.Label>wp</Form.Check.Label>
                                </WithTooltip>
                            </Form.Check>
                            <Form.Check id="phc" inline custom className="record-type-checkbox">
                                <Form.Check.Input type="checkbox"
                                    onChange={this.recordTypeToggleChange}
                                    checked={this.state.recordTypes.phc} isInvalid={!this.state.isValid} />
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
                            min="5" max="60" className="max-guessing-time-range-slider"
                            onChange={this.maxGuessingTimeChange}
                        />
                    </Form.Group>
                    <div className="text-center mt-4">
                        <Button size="lg" className="ready-btn" onClick={this.onReady}>
                            ready!
                        </Button>    
                    </div> 
                </Form>
            </div>    
        )
    }
}