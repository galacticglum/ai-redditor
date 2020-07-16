import React, { Component, useState } from 'react';
import ConfigPage from './ConfigPage';
import Alert from 'react-bootstrap/Alert';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import './GamePage.css';

import axios from 'axios';
import { API_BASE_URL } from '../../App';

function DismissibleAlert(props) {
    const [show, setShow] = useState(true);

    if (show) {
        return (
            <Alert {...props} onClose={() => setShow(false)} dismissible>
                {props.children}
            </Alert>
        )
    }

    return null;
}

export default class GamePage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            isConfigPageVisible: true,
            hasError: false,
        };
    }

    onConfigPageReady = (configState) => {
        // Map the record types from a (string, bool) key/value pair to
        // an array of strings indicating the types to sample from.
        let recordTypes = [];
        for (const [key, value] of Object.entries(configState.recordTypes)) {
            if (!value) continue;
            recordTypes.push(key);
        }

        this.setState({
            gameConfig: {
                ...configState,
                recordTypes: recordTypes
            }
        }, () => {
            this.nextRecord();
        });
    }

    nextRecord = () => {
        // Randomly select record type
        const recordTypes = this.state.gameConfig.recordTypes;
        const recordType = recordTypes[Math.floor(Math.random() * recordTypes.length)];
        this.fetchRecord(recordType, false);
    }

    fetchRecord = (recordType, isGenerated) => {
        const requestData = {
            // A "custom" record refers to a user-generated record
            'is_custom': false,
            // If a record is not generated, it means it was not 'written'
            // by the GPT2 model (i.e. it is written by a human).
            'is_generated': isGenerated,
        };

        this.setState({hasError: false});
        axios.post(`${API_BASE_URL}/r/${recordType}/random`, {
            ...requestData,    
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            this.setState({
                currentRecord: response.data
            });
        })
        .catch(error => {
            this.setState({hasError: true});
            console.log(error);
        });
    }

    render() {
        return (
            <Container>
                <Row>
                    <Col sm="12" md="8" lg="6" className="mx-auto">
                        <div id="view-wrapper">
                            {
                                (this.state.hasError ? 
                                    (<DismissibleAlert variant="danger" className="error-alert">
                                        <Alert.Heading>server go brrr</Alert.Heading>
                                        This is probably an issue with our servers. Please try again later.
                                    </DismissibleAlert>                )
                                : null)
                            }
                            {
                                (this.state.currentRecord === undefined ? 
                                    (<ConfigPage action={this.onConfigPageReady} />)
                                : null)
                            }
                        </div>    
                    </Col>
                </Row>
            </Container>
        );
    }
}
