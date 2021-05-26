# Notes

How to run a simulated circuit: 
https://quantum-computing.ibm.com/lab/docs/iql/first-circuit#execute-the-experiment 
- first run simulation: 
-       simulator = Aer.get_backend('qasm_simulator')
        result = execute(circuit, backend = simulator).result()
        from qiskit.tools.visualisation import plot_histogram



- Running on IBM quantum: 
-       IBMQ.load_account() 
        provider = ibm.get_provider('ibm-q')
        qcomp = provider.get_backend('ibmq_16melbourne')
        job = execute(circuit, backend = qcomp)
        from qiskit.tools.monitor import job_monitor
        result = job.result 
        plot_histogram(result.get_count)

    NEED to get an API token! 

        from qiskit import IBMQ
        IBMQ.save_account('insert token here')
        ibm.load_account()


    https://subscription.packtpub.com/book/programming/9781838828448/1/ch01lvl1sec06/installing-your-api-key-and-accessing-your-provider



